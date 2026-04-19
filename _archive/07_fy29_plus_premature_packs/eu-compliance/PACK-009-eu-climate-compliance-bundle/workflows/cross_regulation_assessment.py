# -*- coding: utf-8 -*-
"""
Cross-Regulation Assessment Workflow
==========================================

Four-phase workflow that initializes all four constituent regulation packs
(CSRD, CBAM, EU Taxonomy, EUDR), runs each pack's compliance assessment
in parallel (simulated), cross-checks results for consistency across
shared data fields, and consolidates into a unified compliance status.

Phases:
    1. PackInitialization - Initialize config for all 4 packs
    2. ParallelAssessment - Run each pack's assessment, collect results
    3. ConsistencyCheck - Cross-check results using shared data fields
    4. Consolidation - Consolidate into unified compliance status

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class RegulationPack(str, Enum):
    """Constituent regulation packs in the bundle."""
    CSRD = "CSRD"
    CBAM = "CBAM"
    EU_TAXONOMY = "EU_TAXONOMY"
    EUDR = "EUDR"


class ComplianceLevel(str, Enum):
    """Compliance level for a regulation."""
    FULLY_COMPLIANT = "FULLY_COMPLIANT"
    SUBSTANTIALLY_COMPLIANT = "SUBSTANTIALLY_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NOT_ASSESSED = "NOT_ASSESSED"


class ConsistencyStatus(str, Enum):
    """Status of cross-regulation consistency check."""
    CONSISTENT = "CONSISTENT"
    MINOR_DISCREPANCY = "MINOR_DISCREPANCY"
    MAJOR_DISCREPANCY = "MAJOR_DISCREPANCY"
    CRITICAL_CONFLICT = "CRITICAL_CONFLICT"


# =============================================================================
# PACK ASSESSMENT CONFIGURATION
# =============================================================================


PACK_ASSESSMENT_CATEGORIES: Dict[str, List[Dict[str, Any]]] = {
    RegulationPack.CSRD.value: [
        {"category": "E1_climate_change", "weight": 0.25, "mandatory": True, "description": "ESRS E1 Climate Change disclosures"},
        {"category": "E2_pollution", "weight": 0.10, "mandatory": False, "description": "ESRS E2 Pollution disclosures"},
        {"category": "E3_water_marine", "weight": 0.08, "mandatory": False, "description": "ESRS E3 Water and marine resources"},
        {"category": "E4_biodiversity", "weight": 0.08, "mandatory": False, "description": "ESRS E4 Biodiversity and ecosystems"},
        {"category": "E5_circular_economy", "weight": 0.08, "mandatory": False, "description": "ESRS E5 Resource use and circular economy"},
        {"category": "S1_own_workforce", "weight": 0.12, "mandatory": True, "description": "ESRS S1 Own workforce"},
        {"category": "S2_value_chain_workers", "weight": 0.05, "mandatory": False, "description": "ESRS S2 Workers in value chain"},
        {"category": "G1_governance", "weight": 0.12, "mandatory": True, "description": "ESRS G1 Business conduct"},
        {"category": "general_disclosures", "weight": 0.07, "mandatory": True, "description": "ESRS 2 General disclosures"},
        {"category": "double_materiality", "weight": 0.05, "mandatory": True, "description": "Double materiality assessment"},
    ],
    RegulationPack.CBAM.value: [
        {"category": "import_registration", "weight": 0.15, "mandatory": True, "description": "Authorized declarant registration"},
        {"category": "goods_classification", "weight": 0.15, "mandatory": True, "description": "CN code classification of goods"},
        {"category": "embedded_emissions_calc", "weight": 0.20, "mandatory": True, "description": "Embedded emissions calculation"},
        {"category": "supplier_verification", "weight": 0.15, "mandatory": True, "description": "Supplier emissions data verification"},
        {"category": "certificate_management", "weight": 0.10, "mandatory": True, "description": "CBAM certificate purchase/surrender"},
        {"category": "quarterly_reporting", "weight": 0.10, "mandatory": True, "description": "Quarterly CBAM reports"},
        {"category": "annual_declaration", "weight": 0.10, "mandatory": True, "description": "Annual CBAM declaration"},
        {"category": "carbon_price_deduction", "weight": 0.05, "mandatory": False, "description": "Carbon price paid in origin country"},
    ],
    RegulationPack.EU_TAXONOMY.value: [
        {"category": "eligibility_screening", "weight": 0.15, "mandatory": True, "description": "Taxonomy eligibility assessment"},
        {"category": "substantial_contribution", "weight": 0.20, "mandatory": True, "description": "Substantial contribution to objectives"},
        {"category": "dnsh_climate_mitigation", "weight": 0.12, "mandatory": True, "description": "DNSH climate change mitigation"},
        {"category": "dnsh_climate_adaptation", "weight": 0.10, "mandatory": True, "description": "DNSH climate change adaptation"},
        {"category": "dnsh_water", "weight": 0.05, "mandatory": False, "description": "DNSH water and marine resources"},
        {"category": "dnsh_circular_economy", "weight": 0.05, "mandatory": False, "description": "DNSH circular economy"},
        {"category": "dnsh_pollution", "weight": 0.05, "mandatory": False, "description": "DNSH pollution prevention"},
        {"category": "dnsh_biodiversity", "weight": 0.05, "mandatory": False, "description": "DNSH biodiversity and ecosystems"},
        {"category": "minimum_safeguards", "weight": 0.10, "mandatory": True, "description": "Minimum social safeguards"},
        {"category": "kpi_reporting", "weight": 0.13, "mandatory": True, "description": "Revenue/CapEx/OpEx KPI reporting"},
    ],
    RegulationPack.EUDR.value: [
        {"category": "commodity_identification", "weight": 0.12, "mandatory": True, "description": "Regulated commodity identification"},
        {"category": "geolocation_tracking", "weight": 0.15, "mandatory": True, "description": "Plot-level geolocation data"},
        {"category": "deforestation_assessment", "weight": 0.18, "mandatory": True, "description": "Deforestation-free status assessment"},
        {"category": "legality_verification", "weight": 0.12, "mandatory": True, "description": "Legality of production verification"},
        {"category": "supply_chain_mapping", "weight": 0.15, "mandatory": True, "description": "Full supply chain traceability"},
        {"category": "risk_assessment", "weight": 0.10, "mandatory": True, "description": "Country and supplier risk assessment"},
        {"category": "due_diligence_statement", "weight": 0.10, "mandatory": True, "description": "Due diligence statement preparation"},
        {"category": "monitoring_system", "weight": 0.08, "mandatory": False, "description": "Ongoing monitoring system"},
    ],
}

# Shared data fields used in cross-regulation consistency checks
SHARED_ASSESSMENT_FIELDS: Dict[str, List[str]] = {
    "ghg_emissions_scope1": [RegulationPack.CSRD.value, RegulationPack.CBAM.value, RegulationPack.EU_TAXONOMY.value],
    "ghg_emissions_scope2": [RegulationPack.CSRD.value, RegulationPack.CBAM.value, RegulationPack.EU_TAXONOMY.value],
    "energy_consumption": [RegulationPack.CSRD.value, RegulationPack.EU_TAXONOMY.value],
    "supply_chain_traceability": [RegulationPack.CBAM.value, RegulationPack.EUDR.value],
    "country_of_origin_data": [RegulationPack.CBAM.value, RegulationPack.EUDR.value],
    "carbon_pricing_data": [RegulationPack.CSRD.value, RegulationPack.CBAM.value],
    "biodiversity_impact": [RegulationPack.CSRD.value, RegulationPack.EUDR.value, RegulationPack.EU_TAXONOMY.value],
    "water_usage": [RegulationPack.CSRD.value, RegulationPack.EU_TAXONOMY.value],
    "governance_safeguards": [RegulationPack.CSRD.value, RegulationPack.EU_TAXONOMY.value],
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class PackAssessmentData(BaseModel):
    """Input data for a single pack's assessment."""
    organization_id: str = Field(...)
    reporting_year: int = Field(...)
    data_points: Dict[str, Any] = Field(default_factory=dict)
    overrides: Dict[str, Any] = Field(default_factory=dict)


class WorkflowConfig(BaseModel):
    """Configuration for cross-regulation assessment workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2024, le=2050)
    target_packs: List[RegulationPack] = Field(
        default_factory=lambda: list(RegulationPack)
    )
    assessment_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Shared data used in assessments"
    )
    pack_overrides: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-pack configuration overrides"
    )
    consistency_threshold: float = Field(
        default=0.05,
        ge=0.0, le=1.0,
        description="Maximum allowed variance for consistency checks"
    )
    skip_phases: List[str] = Field(default_factory=list)


class CrossRegulationAssessmentResult(WorkflowResult):
    """Result from cross-regulation assessment workflow."""
    packs_assessed: int = Field(default=0)
    overall_compliance_level: str = Field(default="NOT_ASSESSED")
    consistency_issues: int = Field(default=0)
    categories_evaluated: int = Field(default=0)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CrossRegulationAssessmentWorkflow:
    """
    Four-phase cross-regulation assessment workflow.

    Initializes all constituent packs, runs compliance assessments,
    cross-checks for consistency, and produces a consolidated status.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = CrossRegulationAssessmentWorkflow()
        >>> config = WorkflowConfig(
        ...     organization_id="org-123",
        ...     reporting_year=2026,
        ...     assessment_data={"ghg_scope1": 15000.0}
        ... )
        >>> result = wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "cross_regulation_assessment"

    PHASE_ORDER = [
        "pack_initialization",
        "parallel_assessment",
        "consistency_check",
        "consolidation",
    ]

    def __init__(self) -> None:
        """Initialize the cross-regulation assessment workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._phase_outputs: Dict[str, Dict[str, Any]] = {}

    def execute(self, config: WorkflowConfig) -> CrossRegulationAssessmentResult:
        """
        Execute the four-phase cross-regulation assessment workflow.

        Args:
            config: Validated workflow configuration.

        Returns:
            CrossRegulationAssessmentResult with assessment outcomes.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting cross-regulation assessment %s for org=%s year=%d",
            self.workflow_id, config.organization_id, config.reporting_year,
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING
        phase_methods = {
            "pack_initialization": self._phase_pack_initialization,
            "parallel_assessment": self._phase_parallel_assessment,
            "consistency_check": self._phase_consistency_check,
            "consolidation": self._phase_consolidation,
        }

        for phase_name in self.PHASE_ORDER:
            if phase_name in config.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                continue

            try:
                phase_result = phase_methods[phase_name](config)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    self._phase_outputs[phase_name] = phase_result.outputs
                elif phase_result.status == PhaseStatus.FAILED:
                    overall_status = WorkflowStatus.FAILED
                    break
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = datetime.utcnow()
        summary = self._build_summary()
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        return CrossRegulationAssessmentResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            packs_assessed=summary.get("packs_assessed", 0),
            overall_compliance_level=summary.get("overall_compliance_level", "NOT_ASSESSED"),
            consistency_issues=summary.get("consistency_issues", 0),
            categories_evaluated=summary.get("categories_evaluated", 0),
        )

    # -------------------------------------------------------------------------
    # Phase 1: Pack Initialization
    # -------------------------------------------------------------------------

    def _phase_pack_initialization(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 1: Initialize configuration for all 4 constituent packs.

        Validates that each pack has the required data and configuration,
        merges any per-pack overrides, and prepares assessment contexts.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            pack_configs: Dict[str, Dict[str, Any]] = {}

            for pack in config.target_packs:
                pack_name = pack.value
                categories = PACK_ASSESSMENT_CATEGORIES.get(pack_name, [])
                overrides = config.pack_overrides.get(pack_name, {})

                mandatory_categories = [c for c in categories if c["mandatory"]]
                optional_categories = [c for c in categories if not c["mandatory"]]

                pack_config = {
                    "pack": pack_name,
                    "organization_id": config.organization_id,
                    "reporting_year": config.reporting_year,
                    "total_categories": len(categories),
                    "mandatory_categories": len(mandatory_categories),
                    "optional_categories": len(optional_categories),
                    "categories": categories,
                    "assessment_data": config.assessment_data,
                    "overrides": overrides,
                    "initialized_at": datetime.utcnow().isoformat(),
                    "status": "INITIALIZED",
                }
                pack_configs[pack_name] = pack_config

                if not categories:
                    warnings.append(f"{pack_name}: no assessment categories defined")

            outputs["pack_configs"] = pack_configs
            outputs["packs_initialized"] = len(pack_configs)
            outputs["total_categories"] = sum(
                pc["total_categories"] for pc in pack_configs.values()
            )
            outputs["total_mandatory"] = sum(
                pc["mandatory_categories"] for pc in pack_configs.values()
            )

            logger.info(
                "Pack initialization complete: %d packs, %d total categories",
                len(pack_configs), outputs["total_categories"],
            )

            status = PhaseStatus.COMPLETED
            records = len(pack_configs)

        except Exception as exc:
            logger.error("Pack initialization failed: %s", exc, exc_info=True)
            errors.append(f"Pack initialization failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="pack_initialization",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Parallel Assessment
    # -------------------------------------------------------------------------

    def _phase_parallel_assessment(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 2: Run each pack's compliance assessment and collect results.

        Simulates running all 4 pack assessments. Each assessment evaluates
        the pack's categories and produces a compliance score.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            init_outputs = self._phase_outputs.get("pack_initialization", {})
            pack_configs = init_outputs.get("pack_configs", {})

            assessment_results: Dict[str, Dict[str, Any]] = {}

            for pack_name, pack_config in pack_configs.items():
                try:
                    result = self._run_pack_assessment(pack_name, pack_config, config)
                    assessment_results[pack_name] = result
                    logger.info(
                        "Assessment for %s complete: score=%.2f level=%s",
                        pack_name, result["overall_score"], result["compliance_level"],
                    )
                except Exception as exc:
                    logger.error("Assessment for %s failed: %s", pack_name, exc)
                    assessment_results[pack_name] = {
                        "pack": pack_name,
                        "overall_score": 0.0,
                        "compliance_level": ComplianceLevel.NOT_ASSESSED.value,
                        "error": str(exc),
                        "category_results": [],
                    }
                    warnings.append(f"Assessment for {pack_name} failed: {str(exc)}")

            outputs["assessment_results"] = assessment_results
            outputs["packs_assessed"] = len(assessment_results)
            outputs["categories_evaluated"] = sum(
                len(r.get("category_results", []))
                for r in assessment_results.values()
            )

            pack_scores = {
                pack: r["overall_score"]
                for pack, r in assessment_results.items()
                if "error" not in r
            }
            outputs["pack_scores"] = pack_scores
            outputs["average_score"] = (
                sum(pack_scores.values()) / max(len(pack_scores), 1)
            )

            status = PhaseStatus.COMPLETED
            records = len(assessment_results)

        except Exception as exc:
            logger.error("Parallel assessment failed: %s", exc, exc_info=True)
            errors.append(f"Parallel assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="parallel_assessment",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _run_pack_assessment(
        self,
        pack_name: str,
        pack_config: Dict[str, Any],
        workflow_config: WorkflowConfig,
    ) -> Dict[str, Any]:
        """
        Run a single pack's compliance assessment (simulated).

        Evaluates each category, computes weighted score,
        and determines compliance level.
        """
        categories = pack_config.get("categories", [])
        assessment_data = pack_config.get("assessment_data", {})

        category_results: List[Dict[str, Any]] = []
        weighted_score_sum = 0.0
        weight_sum = 0.0

        for category in categories:
            cat_name = category["category"]
            weight = category["weight"]
            mandatory = category["mandatory"]

            score = self._evaluate_category(
                pack_name, cat_name, assessment_data, mandatory
            )
            cat_level = self._score_to_compliance_level(score)

            category_results.append({
                "category": cat_name,
                "description": category["description"],
                "weight": weight,
                "mandatory": mandatory,
                "score": round(score, 4),
                "compliance_level": cat_level,
                "assessed_at": datetime.utcnow().isoformat(),
            })

            weighted_score_sum += score * weight
            weight_sum += weight

        overall_score = weighted_score_sum / max(weight_sum, 0.001)
        overall_score = round(overall_score, 4)

        mandatory_scores = [
            cr["score"] for cr in category_results if cr["mandatory"]
        ]
        min_mandatory_score = min(mandatory_scores) if mandatory_scores else 0.0

        if min_mandatory_score < 0.3:
            compliance_level = ComplianceLevel.NON_COMPLIANT.value
        elif overall_score >= 0.85:
            compliance_level = ComplianceLevel.FULLY_COMPLIANT.value
        elif overall_score >= 0.65:
            compliance_level = ComplianceLevel.SUBSTANTIALLY_COMPLIANT.value
        elif overall_score >= 0.40:
            compliance_level = ComplianceLevel.PARTIALLY_COMPLIANT.value
        else:
            compliance_level = ComplianceLevel.NON_COMPLIANT.value

        return {
            "pack": pack_name,
            "overall_score": overall_score,
            "compliance_level": compliance_level,
            "category_results": category_results,
            "categories_assessed": len(category_results),
            "mandatory_min_score": round(min_mandatory_score, 4),
            "assessed_at": datetime.utcnow().isoformat(),
        }

    def _evaluate_category(
        self,
        pack_name: str,
        category: str,
        assessment_data: Dict[str, Any],
        mandatory: bool,
    ) -> float:
        """
        Evaluate a single assessment category (simulated).

        Uses a deterministic simulation based on presence of
        relevant data fields in the assessment data.
        """
        category_field_map: Dict[str, List[str]] = {
            "E1_climate_change": ["ghg_scope1", "ghg_scope2", "ghg_scope3", "climate_transition_plan"],
            "E2_pollution": ["pollution_data", "air_quality"],
            "E3_water_marine": ["water_consumption", "water_discharge"],
            "E4_biodiversity": ["biodiversity_assessment", "land_use"],
            "E5_circular_economy": ["waste_generated", "recycling_rate"],
            "S1_own_workforce": ["employee_count", "workforce_diversity"],
            "S2_value_chain_workers": ["supply_chain_workers", "labor_standards"],
            "G1_governance": ["governance_structure", "ethics_policy"],
            "general_disclosures": ["organization_id", "reporting_year"],
            "double_materiality": ["materiality_assessment"],
            "import_registration": ["declarant_id", "authorization_status"],
            "goods_classification": ["cn_codes", "goods_categories"],
            "embedded_emissions_calc": ["ghg_scope1", "ghg_scope2", "production_volume"],
            "supplier_verification": ["supplier_data", "verification_status"],
            "certificate_management": ["certificates_purchased", "certificates_surrendered"],
            "quarterly_reporting": ["quarterly_report_status"],
            "annual_declaration": ["annual_declaration_status"],
            "carbon_price_deduction": ["carbon_price_origin"],
            "eligibility_screening": ["taxonomy_activities", "nace_codes"],
            "substantial_contribution": ["contribution_assessment"],
            "dnsh_climate_mitigation": ["ghg_scope1", "ghg_scope2"],
            "dnsh_climate_adaptation": ["adaptation_assessment"],
            "dnsh_water": ["water_consumption"],
            "dnsh_circular_economy": ["waste_generated"],
            "dnsh_pollution": ["pollution_data"],
            "dnsh_biodiversity": ["biodiversity_assessment"],
            "minimum_safeguards": ["human_rights_policy", "anti_corruption"],
            "kpi_reporting": ["revenue", "capex", "opex"],
            "commodity_identification": ["commodity_types"],
            "geolocation_tracking": ["geolocations"],
            "deforestation_assessment": ["deforestation_status", "satellite_data"],
            "legality_verification": ["legality_docs"],
            "supply_chain_mapping": ["supply_chain_map", "supplier_data"],
            "risk_assessment": ["risk_scores", "country_risk"],
            "due_diligence_statement": ["dd_statement_status"],
            "monitoring_system": ["monitoring_active"],
        }

        relevant_fields = category_field_map.get(category, [])
        if not relevant_fields:
            return 0.5

        present_count = sum(
            1 for f in relevant_fields if f in assessment_data
        )
        data_coverage = present_count / max(len(relevant_fields), 1)

        base_score = data_coverage * 0.7

        if mandatory:
            base_score += 0.15 if data_coverage > 0 else 0.0
        else:
            base_score += 0.10 if data_coverage > 0 else 0.0

        category_hash = hashlib.md5(
            f"{pack_name}:{category}".encode()
        ).hexdigest()
        variation = (int(category_hash[:4], 16) % 100) / 1000.0
        base_score += variation

        return min(max(base_score, 0.0), 1.0)

    def _score_to_compliance_level(self, score: float) -> str:
        """Convert numeric score to compliance level."""
        if score >= 0.85:
            return ComplianceLevel.FULLY_COMPLIANT.value
        elif score >= 0.65:
            return ComplianceLevel.SUBSTANTIALLY_COMPLIANT.value
        elif score >= 0.40:
            return ComplianceLevel.PARTIALLY_COMPLIANT.value
        else:
            return ComplianceLevel.NON_COMPLIANT.value

    # -------------------------------------------------------------------------
    # Phase 3: Consistency Check
    # -------------------------------------------------------------------------

    def _phase_consistency_check(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 3: Cross-check assessment results for consistency.

        Compares results across packs using shared data fields to ensure
        the same underlying data produces consistent assessment outcomes.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            assessment_outputs = self._phase_outputs.get("parallel_assessment", {})
            assessment_results = assessment_outputs.get("assessment_results", {})
            threshold = config.consistency_threshold

            consistency_checks: List[Dict[str, Any]] = []

            for field_name, packs in SHARED_ASSESSMENT_FIELDS.items():
                involved_packs = [
                    p for p in packs if p in assessment_results
                ]
                if len(involved_packs) < 2:
                    continue

                field_scores: Dict[str, float] = {}
                for pack_name in involved_packs:
                    pack_result = assessment_results[pack_name]
                    cat_results = pack_result.get("category_results", [])
                    related_cats = self._find_related_categories(
                        pack_name, field_name, cat_results
                    )
                    if related_cats:
                        avg_score = sum(c["score"] for c in related_cats) / len(related_cats)
                        field_scores[pack_name] = avg_score

                if len(field_scores) < 2:
                    continue

                scores = list(field_scores.values())
                max_variance = max(scores) - min(scores)

                if max_variance <= threshold * 0.5:
                    status_val = ConsistencyStatus.CONSISTENT.value
                elif max_variance <= threshold:
                    status_val = ConsistencyStatus.MINOR_DISCREPANCY.value
                elif max_variance <= threshold * 2:
                    status_val = ConsistencyStatus.MAJOR_DISCREPANCY.value
                else:
                    status_val = ConsistencyStatus.CRITICAL_CONFLICT.value

                check_result = {
                    "field": field_name,
                    "packs_compared": involved_packs,
                    "scores": field_scores,
                    "variance": round(max_variance, 4),
                    "threshold": threshold,
                    "status": status_val,
                    "checked_at": datetime.utcnow().isoformat(),
                }
                consistency_checks.append(check_result)

                if status_val == ConsistencyStatus.CRITICAL_CONFLICT.value:
                    errors.append(
                        f"Critical conflict on '{field_name}': "
                        f"variance={max_variance:.4f} exceeds 2x threshold"
                    )
                elif status_val == ConsistencyStatus.MAJOR_DISCREPANCY.value:
                    warnings.append(
                        f"Major discrepancy on '{field_name}': "
                        f"variance={max_variance:.4f}"
                    )

            outputs["consistency_checks"] = consistency_checks
            outputs["total_checks"] = len(consistency_checks)
            outputs["consistent_count"] = sum(
                1 for c in consistency_checks
                if c["status"] == ConsistencyStatus.CONSISTENT.value
            )
            outputs["minor_discrepancies"] = sum(
                1 for c in consistency_checks
                if c["status"] == ConsistencyStatus.MINOR_DISCREPANCY.value
            )
            outputs["major_discrepancies"] = sum(
                1 for c in consistency_checks
                if c["status"] == ConsistencyStatus.MAJOR_DISCREPANCY.value
            )
            outputs["critical_conflicts"] = sum(
                1 for c in consistency_checks
                if c["status"] == ConsistencyStatus.CRITICAL_CONFLICT.value
            )
            outputs["overall_consistent"] = outputs["critical_conflicts"] == 0

            logger.info(
                "Consistency check complete: %d checks, %d consistent, %d issues",
                len(consistency_checks), outputs["consistent_count"],
                outputs["major_discrepancies"] + outputs["critical_conflicts"],
            )

            status = PhaseStatus.COMPLETED
            records = len(consistency_checks)

        except Exception as exc:
            logger.error("Consistency check failed: %s", exc, exc_info=True)
            errors.append(f"Consistency check failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="consistency_check",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _find_related_categories(
        self,
        pack_name: str,
        field_name: str,
        category_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find assessment categories related to a shared field."""
        field_category_map: Dict[str, Dict[str, List[str]]] = {
            "ghg_emissions_scope1": {
                RegulationPack.CSRD.value: ["E1_climate_change"],
                RegulationPack.CBAM.value: ["embedded_emissions_calc"],
                RegulationPack.EU_TAXONOMY.value: ["dnsh_climate_mitigation"],
            },
            "ghg_emissions_scope2": {
                RegulationPack.CSRD.value: ["E1_climate_change"],
                RegulationPack.CBAM.value: ["embedded_emissions_calc"],
                RegulationPack.EU_TAXONOMY.value: ["dnsh_climate_mitigation"],
            },
            "energy_consumption": {
                RegulationPack.CSRD.value: ["E1_climate_change"],
                RegulationPack.EU_TAXONOMY.value: ["substantial_contribution"],
            },
            "supply_chain_traceability": {
                RegulationPack.CBAM.value: ["supplier_verification"],
                RegulationPack.EUDR.value: ["supply_chain_mapping"],
            },
            "country_of_origin_data": {
                RegulationPack.CBAM.value: ["goods_classification"],
                RegulationPack.EUDR.value: ["geolocation_tracking"],
            },
            "carbon_pricing_data": {
                RegulationPack.CSRD.value: ["E1_climate_change"],
                RegulationPack.CBAM.value: ["certificate_management"],
            },
            "biodiversity_impact": {
                RegulationPack.CSRD.value: ["E4_biodiversity"],
                RegulationPack.EUDR.value: ["deforestation_assessment"],
                RegulationPack.EU_TAXONOMY.value: ["dnsh_biodiversity"],
            },
            "water_usage": {
                RegulationPack.CSRD.value: ["E3_water_marine"],
                RegulationPack.EU_TAXONOMY.value: ["dnsh_water"],
            },
            "governance_safeguards": {
                RegulationPack.CSRD.value: ["G1_governance"],
                RegulationPack.EU_TAXONOMY.value: ["minimum_safeguards"],
            },
        }

        target_cats = field_category_map.get(field_name, {}).get(pack_name, [])
        return [
            cr for cr in category_results if cr["category"] in target_cats
        ]

    # -------------------------------------------------------------------------
    # Phase 4: Consolidation
    # -------------------------------------------------------------------------

    def _phase_consolidation(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 4: Consolidate into unified compliance status.

        Aggregates results from all packs, factors in consistency
        findings, and produces a single bundle-level compliance status.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            assessment_outputs = self._phase_outputs.get("parallel_assessment", {})
            consistency_outputs = self._phase_outputs.get("consistency_check", {})
            assessment_results = assessment_outputs.get("assessment_results", {})
            overall_consistent = consistency_outputs.get("overall_consistent", True)

            pack_summaries: List[Dict[str, Any]] = []
            for pack_name, result in assessment_results.items():
                pack_summaries.append({
                    "pack": pack_name,
                    "score": result.get("overall_score", 0.0),
                    "compliance_level": result.get("compliance_level", "NOT_ASSESSED"),
                    "categories_assessed": result.get("categories_assessed", 0),
                    "mandatory_min_score": result.get("mandatory_min_score", 0.0),
                })

            pack_scores = [s["score"] for s in pack_summaries]
            bundle_score = sum(pack_scores) / max(len(pack_scores), 1)
            min_pack_score = min(pack_scores) if pack_scores else 0.0

            if not overall_consistent:
                bundle_score *= 0.9

            bundle_score = round(bundle_score, 4)

            if min_pack_score < 0.3:
                bundle_compliance = ComplianceLevel.NON_COMPLIANT.value
            elif bundle_score >= 0.85:
                bundle_compliance = ComplianceLevel.FULLY_COMPLIANT.value
            elif bundle_score >= 0.65:
                bundle_compliance = ComplianceLevel.SUBSTANTIALLY_COMPLIANT.value
            elif bundle_score >= 0.40:
                bundle_compliance = ComplianceLevel.PARTIALLY_COMPLIANT.value
            else:
                bundle_compliance = ComplianceLevel.NON_COMPLIANT.value

            priority_actions: List[Dict[str, Any]] = []
            for summary in pack_summaries:
                if summary["compliance_level"] in (
                    ComplianceLevel.NON_COMPLIANT.value,
                    ComplianceLevel.PARTIALLY_COMPLIANT.value,
                ):
                    priority_actions.append({
                        "pack": summary["pack"],
                        "current_level": summary["compliance_level"],
                        "score": summary["score"],
                        "action": f"Improve {summary['pack']} compliance from {summary['score']:.2f} to 0.65+",
                        "priority": "HIGH" if summary["score"] < 0.40 else "MEDIUM",
                    })

            outputs["pack_summaries"] = pack_summaries
            outputs["bundle_score"] = bundle_score
            outputs["bundle_compliance_level"] = bundle_compliance
            outputs["min_pack_score"] = round(min_pack_score, 4)
            outputs["consistency_applied"] = not overall_consistent
            outputs["priority_actions"] = priority_actions
            outputs["priority_action_count"] = len(priority_actions)
            outputs["consolidated_at"] = datetime.utcnow().isoformat()

            outputs["regulatory_dashboard"] = {
                "organization_id": config.organization_id,
                "reporting_year": config.reporting_year,
                "bundle_compliance": bundle_compliance,
                "bundle_score": bundle_score,
                "pack_compliance": {
                    s["pack"]: s["compliance_level"] for s in pack_summaries
                },
                "needs_attention": len(priority_actions) > 0,
                "generated_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Consolidation complete: bundle_score=%.4f level=%s actions=%d",
                bundle_score, bundle_compliance, len(priority_actions),
            )

            status = PhaseStatus.COMPLETED
            records = len(pack_summaries)

        except Exception as exc:
            logger.error("Consolidation failed: %s", exc, exc_info=True)
            errors.append(f"Consolidation failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="consolidation",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build workflow summary from all phase outputs."""
        init_out = self._phase_outputs.get("pack_initialization", {})
        assess_out = self._phase_outputs.get("parallel_assessment", {})
        consistency_out = self._phase_outputs.get("consistency_check", {})
        consolidation_out = self._phase_outputs.get("consolidation", {})

        return {
            "packs_initialized": init_out.get("packs_initialized", 0),
            "packs_assessed": assess_out.get("packs_assessed", 0),
            "categories_evaluated": assess_out.get("categories_evaluated", 0),
            "average_score": assess_out.get("average_score", 0.0),
            "consistency_issues": (
                consistency_out.get("major_discrepancies", 0)
                + consistency_out.get("critical_conflicts", 0)
            ),
            "overall_compliance_level": consolidation_out.get(
                "bundle_compliance_level", "NOT_ASSESSED"
            ),
            "bundle_score": consolidation_out.get("bundle_score", 0.0),
            "priority_actions": consolidation_out.get("priority_action_count", 0),
        }


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
