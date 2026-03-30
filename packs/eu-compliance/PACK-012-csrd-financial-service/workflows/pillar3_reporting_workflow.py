# -*- coding: utf-8 -*-
"""
Pillar 3 ESG Reporting Workflow
===================================

Four-phase workflow for generating EBA Pillar 3 ESG ITS disclosures for
credit institutions under CRR3/CRD6.

Phases:
    1. DataExtraction - Extract data from internal systems for Pillar 3 templates
    2. TemplatePopulation - Populate EBA ITS quantitative and qualitative templates
    3. QualityValidation - Validate data consistency and completeness
    4. FilingPreparation - Prepare final filing package for supervisory submission

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

class PhaseStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

class WorkflowContext(BaseModel):
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED

class PhaseResult(BaseModel):
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)

class WorkflowResult(BaseModel):
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
#  Input / Result Models
# ---------------------------------------------------------------------------

class Pillar3DataRecord(BaseModel):
    """Data record for Pillar 3 ESG template."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_id: str = Field(..., description="EBA template ID (e.g. Template_1)")
    row_id: str = Field(default="", description="Template row reference")
    column_id: str = Field(default="", description="Template column reference")
    value: Optional[float] = Field(None)
    text_value: Optional[str] = Field(None)
    sector: str = Field(default="")
    maturity_bucket: str = Field(default="")
    geography: str = Field(default="")

class Pillar3ReportingInput(BaseModel):
    """Input for the Pillar 3 reporting workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_date: str = Field(..., description="Reporting date YYYY-MM-DD")
    institution_name: str = Field(default="")
    lei: str = Field(default="")
    data_records: List[Pillar3DataRecord] = Field(default_factory=list)
    gar_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    btar_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    financed_emissions_tco2e: Optional[float] = Field(None, ge=0.0)
    total_exposure_eur: Optional[float] = Field(None, ge=0.0)
    top_20_carbon_intensive: List[Dict[str, Any]] = Field(default_factory=list)
    qualitative_disclosures: Dict[str, str] = Field(default_factory=dict)
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v

class Pillar3ReportingResult(WorkflowResult):
    """Result from the Pillar 3 reporting workflow."""
    templates_populated: int = Field(default=0)
    templates_total: int = Field(default=10)
    data_quality_score: float = Field(default=0.0)
    validation_passed: bool = Field(default=False)
    filing_ready: bool = Field(default=False)
    issues_count: int = Field(default=0)
    completeness_pct: float = Field(default=0.0)

# ---------------------------------------------------------------------------
#  Phases
# ---------------------------------------------------------------------------

class DataExtractionPhase:
    """Extract and organize data for Pillar 3 ESG templates."""
    PHASE_NAME = "data_extraction"

    EBA_TEMPLATES = [
        "Template_1_Banking_Book_CC_Transition",
        "Template_2_Banking_Book_CC_Physical",
        "Template_3_Real_Estate_EPC",
        "Template_4_Alignment_Metrics",
        "Template_5_Exposures_Top20",
        "Template_6_Trading_Book",
        "Template_7_ESG_Risks_Banking_Book",
        "Template_8_Qualitative_ESG",
        "Template_9_Mitigating_Actions",
        "Template_10_KPIs",
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            config = context.config
            records = config.get("data_records", [])

            by_template: Dict[str, List[Dict[str, Any]]] = {}
            for r in records:
                tid = r.get("template_id", "UNKNOWN")
                by_template.setdefault(tid, []).append(r)

            outputs["data_by_template"] = by_template
            outputs["templates_with_data"] = list(by_template.keys())
            outputs["templates_with_data_count"] = len(by_template)
            outputs["total_records"] = len(records)

            # Enrich with high-level KPIs
            outputs["gar_pct"] = config.get("gar_pct")
            outputs["btar_pct"] = config.get("btar_pct")
            outputs["financed_emissions_tco2e"] = config.get("financed_emissions_tco2e")
            outputs["total_exposure_eur"] = config.get("total_exposure_eur")
            outputs["top_20_carbon_intensive"] = config.get("top_20_carbon_intensive", [])

            missing = [t for t in self.EBA_TEMPLATES if t not in by_template]
            if missing:
                warnings.append(f"{len(missing)} EBA templates without data: {', '.join(missing[:3])}...")
            outputs["missing_templates"] = missing

            status = PhaseStatus.COMPLETED
            records_count = len(records)
        except Exception as exc:
            logger.error("DataExtraction failed: %s", exc, exc_info=True)
            errors.append(f"Data extraction failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records_count = 0
        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs), records_processed=records_count,
        )

class TemplatePopulationPhase:
    """Populate EBA ITS quantitative and qualitative templates."""
    PHASE_NAME = "template_population"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            extraction = context.get_phase_output("data_extraction")
            config = context.config
            by_template = extraction.get("data_by_template", {})

            populated = {}
            count = 0

            # Template 4: Alignment metrics
            populated["Template_4_Alignment_Metrics"] = {
                "gar_pct": extraction.get("gar_pct"),
                "btar_pct": extraction.get("btar_pct"),
                "financed_emissions": extraction.get("financed_emissions_tco2e"),
            }
            count += 1

            # Template 5: Top 20
            top20 = extraction.get("top_20_carbon_intensive", [])
            populated["Template_5_Exposures_Top20"] = {"top_20": top20}
            count += 1

            # Template 8: Qualitative
            qual = config.get("qualitative_disclosures", {})
            populated["Template_8_Qualitative_ESG"] = qual
            count += 1

            # Template 10: KPIs
            populated["Template_10_KPIs"] = {
                "gar": extraction.get("gar_pct"),
                "financed_emissions": extraction.get("financed_emissions_tco2e"),
                "total_exposure": extraction.get("total_exposure_eur"),
            }
            count += 1

            # Populate remaining from data records
            for tid, records in by_template.items():
                if tid not in populated:
                    populated[tid] = {"records": records, "record_count": len(records)}
                    count += 1

            outputs["populated_templates"] = populated
            outputs["templates_populated"] = count
            outputs["templates_total"] = 10

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("TemplatePopulation failed: %s", exc, exc_info=True)
            errors.append(f"Template population failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class QualityValidationPhase:
    """Validate data consistency and completeness for Pillar 3."""
    PHASE_NAME = "quality_validation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            population = context.get_phase_output("template_population")
            populated = population.get("populated_templates", {})
            total = population.get("templates_total", 10)
            count = population.get("templates_populated", 0)

            issues = []
            completeness = round(count / max(total, 1) * 100, 2)

            if completeness < 100:
                issues.append({"severity": "high", "issue": f"Only {count}/{total} templates populated"})
            # Check key metrics exist
            t4 = populated.get("Template_4_Alignment_Metrics", {})
            if t4.get("gar_pct") is None:
                issues.append({"severity": "medium", "issue": "GAR not computed"})
            if t4.get("financed_emissions") is None:
                issues.append({"severity": "medium", "issue": "Financed emissions not computed"})

            t5 = populated.get("Template_5_Exposures_Top20", {})
            if len(t5.get("top_20", [])) < 20:
                issues.append({"severity": "low", "issue": f"Top 20 has {len(t5.get('top_20', []))} entries"})

            validation_passed = not any(i["severity"] == "high" for i in issues)
            dq_score = round(max(0, 100 - len(issues) * 10), 2)

            outputs["issues"] = issues
            outputs["issues_count"] = len(issues)
            outputs["validation_passed"] = validation_passed
            outputs["data_quality_score"] = dq_score
            outputs["completeness_pct"] = completeness

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("QualityValidation failed: %s", exc, exc_info=True)
            errors.append(f"Quality validation failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class FilingPreparationPhase:
    """Prepare final Pillar 3 ESG filing package."""
    PHASE_NAME = "filing_preparation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            config = context.config
            validation = context.get_phase_output("quality_validation")
            population = context.get_phase_output("template_population")

            filing_ready = validation.get("validation_passed", False)
            outputs["filing_ready"] = filing_ready
            outputs["filing_package"] = {
                "institution_name": config.get("institution_name", ""),
                "lei": config.get("lei", ""),
                "reporting_date": config.get("reporting_date", ""),
                "templates_populated": population.get("templates_populated", 0),
                "completeness_pct": validation.get("completeness_pct", 0.0),
                "data_quality_score": validation.get("data_quality_score", 0.0),
                "issues_count": validation.get("issues_count", 0),
                "generated_at": utcnow().isoformat(),
            }
            if not filing_ready:
                warnings.append("Filing package not ready: validation issues exist")

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("FilingPreparation failed: %s", exc, exc_info=True)
            errors.append(f"Filing preparation failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class Pillar3ReportingWorkflow:
    """Four-phase Pillar 3 ESG reporting workflow for EBA ITS compliance."""

    WORKFLOW_NAME = "pillar3_reporting"
    PHASE_ORDER = ["data_extraction", "template_population",
                    "quality_validation", "filing_preparation"]

    def __init__(self, progress_callback: Optional[Callable[[str, str, float], None]] = None) -> None:
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "data_extraction": DataExtractionPhase(),
            "template_population": TemplatePopulationPhase(),
            "quality_validation": QualityValidationPhase(),
            "filing_preparation": FilingPreparationPhase(),
        }

    async def run(self, input_data: Pillar3ReportingInput) -> Pillar3ReportingResult:
        """Execute the workflow."""
        started_at = utcnow()
        logger.info("Starting %s workflow %s org=%s", self.WORKFLOW_NAME,
                     self.workflow_id, input_data.organization_id)
        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=input_data.model_dump(),
        )
        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                ))
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue
            if context.is_phase_completed(phase_name):
                continue
            self._notify_progress(phase_name, f"Starting: {phase_name}",
                                  idx / len(self.PHASE_ORDER))
            context.mark_phase(phase_name, PhaseStatus.RUNNING)
            try:
                result = await self._phases[phase_name].execute(context)
                completed_phases.append(result)
                if result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, result.status)
                    if phase_name == self.PHASE_ORDER[0]:
                        overall_status = WorkflowStatus.FAILED
                        break
                context.errors.extend(result.errors)
                context.warnings.extend(result.warnings)
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.FAILED,
                    started_at=utcnow(), errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                ))
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                         for p in completed_phases)
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })
        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return Pillar3ReportingResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            **{k: summary.get(k, v) for k, v in self._result_defaults().items()}
        )

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        pop = context.get_phase_output("template_population")
        val = context.get_phase_output("quality_validation")
        filing = context.get_phase_output("filing_preparation")
        return {
            "templates_populated": pop.get("templates_populated", 0),
            "templates_total": pop.get("templates_total", 10),
            "data_quality_score": val.get("data_quality_score", 0.0),
            "validation_passed": val.get("validation_passed", False),
            "filing_ready": filing.get("filing_ready", False),
            "issues_count": val.get("issues_count", 0),
            "completeness_pct": val.get("completeness_pct", 0.0),
        }

    @staticmethod
    def _result_defaults() -> Dict[str, Any]:
        return {
            "templates_populated": 0, "templates_total": 10,
            "data_quality_score": 0.0, "validation_passed": False,
            "filing_ready": False, "issues_count": 0, "completeness_pct": 0.0,
        }
