# -*- coding: utf-8 -*-
"""
AssurancePortalBridge - Assurance Provider Portal Integration for PACK-029
============================================================================

Enterprise bridge for exporting evidence packages to third-party assurance
provider portals. Structures evidence according to ISO 14064-3 workpaper
requirements, manages document uploads (calculation sheets, data sources,
methodology documentation), supports both limited and reasonable assurance
request workflows, integrates with Big 4 provider APIs, validates evidence
completeness, and maintains version control for submitted evidence.

Integration Points:
    - Evidence Package: ISO 14064-3 compliant workpaper structure
    - Document Upload: Calculation sheets, data sources, methodology
    - Assurance Workflows: Limited vs reasonable assurance requests
    - Provider API: Big 4 portal integration (EY, Deloitte, PwC, KPMG)
    - Completeness Checks: Evidence gap analysis
    - Version Control: Submitted evidence versioning and audit trail

Assurance Standards:
    - ISO 14064-3: Specification for validation/verification of GHG assertions
    - ISAE 3000: Assurance Engagements Other than Audits or Reviews
    - ISAE 3410: Assurance Engagements on Greenhouse Gas Statements
    - AA1000 AS: AccountAbility Assurance Standard

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-029 Interim Targets Pack
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

class AssuranceLevel(str, Enum):
    LIMITED = "limited"
    REASONABLE = "reasonable"
    COMBINED = "combined"

class AssuranceStandard(str, Enum):
    ISO_14064_3 = "iso_14064_3"
    ISAE_3000 = "isae_3000"
    ISAE_3410 = "isae_3410"
    AA1000_AS = "aa1000_as"

class AssuranceProvider(str, Enum):
    EY = "ey"
    DELOITTE = "deloitte"
    PWC = "pwc"
    KPMG = "kpmg"
    BDO = "bdo"
    GRANT_THORNTON = "grant_thornton"
    BUREAU_VERITAS = "bureau_veritas"
    SGS = "sgs"
    DNV = "dnv"
    LRQA = "lrqa"
    OTHER = "other"

class DocumentType(str, Enum):
    CALCULATION_SHEET = "calculation_sheet"
    DATA_SOURCE = "data_source"
    METHODOLOGY = "methodology"
    EMISSION_FACTOR = "emission_factor"
    ORGANIZATIONAL_BOUNDARY = "organizational_boundary"
    TARGET_DOCUMENTATION = "target_documentation"
    INTERIM_TARGET_PLAN = "interim_target_plan"
    LINEARITY_EVIDENCE = "linearity_evidence"
    INITIATIVE_EVIDENCE = "initiative_evidence"
    VARIANCE_ANALYSIS = "variance_analysis"
    DATA_QUALITY_REPORT = "data_quality_report"
    AUDIT_TRAIL = "audit_trail"
    SBTI_SUBMISSION = "sbti_submission"
    MANAGEMENT_ASSERTION = "management_assertion"
    SUPPORTING_EVIDENCE = "supporting_evidence"

class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    EVIDENCE_COLLECTION = "evidence_collection"
    COMPLETENESS_CHECK = "completeness_check"
    READY_FOR_SUBMISSION = "ready_for_submission"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    QUERIES_RAISED = "queries_raised"
    QUERIES_RESOLVED = "queries_resolved"
    OPINION_ISSUED = "opinion_issued"
    COMPLETED = "completed"

class EvidenceStatus(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    PROVIDED = "provided"
    MISSING = "missing"
    INCOMPLETE = "incomplete"
    ACCEPTED = "accepted"
    REJECTED = "rejected"

# ---------------------------------------------------------------------------
# ISO 14064-3 Evidence Requirements
# ---------------------------------------------------------------------------

ISO_14064_3_REQUIREMENTS: List[Dict[str, Any]] = [
    {"id": "WP-001", "category": "organizational_boundary", "name": "Organizational boundary definition", "doc_type": "organizational_boundary", "required": True, "assurance_level": "both"},
    {"id": "WP-002", "category": "methodology", "name": "GHG quantification methodology", "doc_type": "methodology", "required": True, "assurance_level": "both"},
    {"id": "WP-003", "category": "base_year", "name": "Base year inventory documentation", "doc_type": "calculation_sheet", "required": True, "assurance_level": "both"},
    {"id": "WP-004", "category": "scope1", "name": "Scope 1 calculation workpapers", "doc_type": "calculation_sheet", "required": True, "assurance_level": "both"},
    {"id": "WP-005", "category": "scope2", "name": "Scope 2 calculation workpapers", "doc_type": "calculation_sheet", "required": True, "assurance_level": "both"},
    {"id": "WP-006", "category": "scope3", "name": "Scope 3 calculation workpapers", "doc_type": "calculation_sheet", "required": True, "assurance_level": "both"},
    {"id": "WP-007", "category": "emission_factors", "name": "Emission factor sources and references", "doc_type": "emission_factor", "required": True, "assurance_level": "both"},
    {"id": "WP-008", "category": "data_sources", "name": "Primary data source documentation", "doc_type": "data_source", "required": True, "assurance_level": "both"},
    {"id": "WP-009", "category": "targets", "name": "Target setting methodology", "doc_type": "target_documentation", "required": True, "assurance_level": "both"},
    {"id": "WP-010", "category": "interim_targets", "name": "Interim target decomposition plan", "doc_type": "interim_target_plan", "required": True, "assurance_level": "both"},
    {"id": "WP-011", "category": "linearity", "name": "Linearity assessment evidence", "doc_type": "linearity_evidence", "required": True, "assurance_level": "both"},
    {"id": "WP-012", "category": "initiatives", "name": "Reduction initiative documentation", "doc_type": "initiative_evidence", "required": True, "assurance_level": "both"},
    {"id": "WP-013", "category": "variance", "name": "Variance analysis and attribution", "doc_type": "variance_analysis", "required": True, "assurance_level": "reasonable"},
    {"id": "WP-014", "category": "data_quality", "name": "Data quality assessment report", "doc_type": "data_quality_report", "required": True, "assurance_level": "reasonable"},
    {"id": "WP-015", "category": "audit_trail", "name": "Calculation audit trail", "doc_type": "audit_trail", "required": True, "assurance_level": "reasonable"},
    {"id": "WP-016", "category": "sbti", "name": "SBTi submission documentation", "doc_type": "sbti_submission", "required": False, "assurance_level": "both"},
    {"id": "WP-017", "category": "management", "name": "Management assertion letter", "doc_type": "management_assertion", "required": True, "assurance_level": "both"},
    {"id": "WP-018", "category": "recalculation", "name": "Base year recalculation policy", "doc_type": "methodology", "required": True, "assurance_level": "both"},
    {"id": "WP-019", "category": "uncertainty", "name": "Uncertainty assessment", "doc_type": "data_quality_report", "required": True, "assurance_level": "reasonable"},
    {"id": "WP-020", "category": "supporting", "name": "Additional supporting evidence", "doc_type": "supporting_evidence", "required": False, "assurance_level": "both"},
]

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class AssurancePortalConfig(BaseModel):
    """Configuration for the assurance portal bridge."""
    pack_id: str = Field(default="PACK-029")
    organization_id: str = Field(default="")
    organization_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    assurance_standard: AssuranceStandard = Field(default=AssuranceStandard.ISAE_3410)
    provider: AssuranceProvider = Field(default=AssuranceProvider.OTHER)
    provider_api_key: str = Field(default="")
    provider_api_url: str = Field(default="")
    enable_provenance: bool = Field(default=True)
    auto_version: bool = Field(default=True)
    max_upload_size_mb: int = Field(default=100, ge=1, le=1000)

class EvidenceDocument(BaseModel):
    """Single evidence document in the assurance package."""
    document_id: str = Field(default_factory=_new_uuid)
    workpaper_ref: str = Field(default="")
    document_type: DocumentType = Field(default=DocumentType.SUPPORTING_EVIDENCE)
    title: str = Field(default="")
    description: str = Field(default="")
    file_name: str = Field(default="")
    file_size_bytes: int = Field(default=0)
    file_hash: str = Field(default="")
    mime_type: str = Field(default="application/pdf")
    version: int = Field(default=1)
    status: EvidenceStatus = Field(default=EvidenceStatus.PROVIDED)
    uploaded_at: datetime = Field(default_factory=utcnow)
    uploaded_by: str = Field(default="")
    reviewer_notes: str = Field(default="")
    provenance_hash: str = Field(default="")

class WorkpaperRequirement(BaseModel):
    """Single workpaper requirement from ISO 14064-3."""
    requirement_id: str = Field(default="")
    category: str = Field(default="")
    name: str = Field(default="")
    document_type: DocumentType = Field(default=DocumentType.SUPPORTING_EVIDENCE)
    required: bool = Field(default=True)
    assurance_level: str = Field(default="both")
    status: EvidenceStatus = Field(default=EvidenceStatus.REQUIRED)
    documents: List[EvidenceDocument] = Field(default_factory=list)

class CompletenessCheck(BaseModel):
    """Evidence completeness check result."""
    check_id: str = Field(default_factory=_new_uuid)
    total_requirements: int = Field(default=0)
    requirements_met: int = Field(default=0)
    requirements_missing: int = Field(default=0)
    requirements_incomplete: int = Field(default=0)
    requirements_optional: int = Field(default=0)
    completeness_pct: float = Field(default=0.0)
    ready_for_submission: bool = Field(default=False)
    gaps: List[Dict[str, str]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class AssuranceRequest(BaseModel):
    """Assurance engagement request."""
    request_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    reporting_year: int = Field(default=2025)
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    assurance_standard: AssuranceStandard = Field(default=AssuranceStandard.ISAE_3410)
    provider: AssuranceProvider = Field(default=AssuranceProvider.OTHER)
    scope_of_engagement: List[str] = Field(default_factory=list)
    status: WorkflowStatus = Field(default=WorkflowStatus.DRAFT)
    evidence_package_id: str = Field(default="")
    submitted_at: Optional[datetime] = Field(None)
    estimated_completion_weeks: int = Field(default=8)
    fee_usd: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class EvidencePackage(BaseModel):
    """Complete evidence package for assurance."""
    package_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    reporting_year: int = Field(default=2025)
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    assurance_standard: AssuranceStandard = Field(default=AssuranceStandard.ISAE_3410)
    workpapers: List[WorkpaperRequirement] = Field(default_factory=list)
    documents: List[EvidenceDocument] = Field(default_factory=list)
    completeness: Optional[CompletenessCheck] = Field(None)
    version: int = Field(default=1)
    status: WorkflowStatus = Field(default=WorkflowStatus.DRAFT)
    created_at: datetime = Field(default_factory=utcnow)
    last_updated: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# AssurancePortalBridge
# ---------------------------------------------------------------------------

class AssurancePortalBridge:
    """Assurance provider portal integration bridge for PACK-029.

    Structures evidence packages according to ISO 14064-3, manages
    document uploads, supports limited/reasonable assurance workflows,
    integrates with Big 4 provider APIs, validates completeness, and
    maintains version control.

    Example:
        >>> bridge = AssurancePortalBridge(AssurancePortalConfig(
        ...     organization_name="Acme Corp",
        ...     assurance_level=AssuranceLevel.LIMITED,
        ...     provider=AssuranceProvider.EY,
        ... ))
        >>> package = await bridge.create_evidence_package()
        >>> await bridge.add_document(package.package_id, doc_data)
        >>> check = await bridge.check_completeness(package.package_id)
        >>> request = await bridge.submit_assurance_request(package.package_id)
    """

    def __init__(self, config: Optional[AssurancePortalConfig] = None) -> None:
        self.config = config or AssurancePortalConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._packages: Dict[str, EvidencePackage] = {}
        self._requests: Dict[str, AssuranceRequest] = {}
        self._http_client: Optional[Any] = None

        self.logger.info(
            "AssurancePortalBridge (PACK-029) initialized: org=%s, "
            "level=%s, standard=%s, provider=%s",
            self.config.organization_name,
            self.config.assurance_level.value,
            self.config.assurance_standard.value,
            self.config.provider.value,
        )

    async def create_evidence_package(self) -> EvidencePackage:
        """Create a new evidence package with ISO 14064-3 workpaper structure."""
        workpapers: List[WorkpaperRequirement] = []

        for req in ISO_14064_3_REQUIREMENTS:
            # Check if applicable for assurance level
            req_level = req.get("assurance_level", "both")
            applicable = (
                req_level == "both"
                or (req_level == "limited" and self.config.assurance_level in (AssuranceLevel.LIMITED, AssuranceLevel.COMBINED))
                or (req_level == "reasonable" and self.config.assurance_level in (AssuranceLevel.REASONABLE, AssuranceLevel.COMBINED))
            )

            if not applicable:
                continue

            wp = WorkpaperRequirement(
                requirement_id=req["id"],
                category=req["category"],
                name=req["name"],
                document_type=DocumentType(req["doc_type"]),
                required=req["required"],
                assurance_level=req_level,
                status=EvidenceStatus.REQUIRED if req["required"] else EvidenceStatus.OPTIONAL,
            )
            workpapers.append(wp)

        package = EvidencePackage(
            organization_name=self.config.organization_name,
            reporting_year=self.config.reporting_year,
            assurance_level=self.config.assurance_level,
            assurance_standard=self.config.assurance_standard,
            workpapers=workpapers,
            version=1,
            status=WorkflowStatus.EVIDENCE_COLLECTION,
        )

        if self.config.enable_provenance:
            package.provenance_hash = _compute_hash(package)

        self._packages[package.package_id] = package
        self.logger.info(
            "Evidence package created: %s, %d workpapers, level=%s",
            package.package_id, len(workpapers), package.assurance_level.value,
        )
        return package

    async def add_document(
        self,
        package_id: str,
        document_data: Dict[str, Any],
    ) -> EvidenceDocument:
        """Add a document to an evidence package."""
        package = self._packages.get(package_id)
        if not package:
            raise ValueError(f"Package {package_id} not found")

        doc = EvidenceDocument(
            workpaper_ref=document_data.get("workpaper_ref", ""),
            document_type=DocumentType(document_data.get("document_type", "supporting_evidence")),
            title=document_data.get("title", ""),
            description=document_data.get("description", ""),
            file_name=document_data.get("file_name", ""),
            file_size_bytes=document_data.get("file_size_bytes", 0),
            file_hash=document_data.get("file_hash", ""),
            mime_type=document_data.get("mime_type", "application/pdf"),
            version=1,
            status=EvidenceStatus.PROVIDED,
            uploaded_by=document_data.get("uploaded_by", ""),
        )

        if self.config.enable_provenance:
            doc.provenance_hash = _compute_hash(doc)

        package.documents.append(doc)

        # Update workpaper status
        for wp in package.workpapers:
            if wp.requirement_id == doc.workpaper_ref:
                wp.documents.append(doc)
                wp.status = EvidenceStatus.PROVIDED
                break

        package.last_updated = utcnow()
        if self.config.auto_version:
            package.version += 1

        self.logger.info(
            "Document added: %s -> package %s, type=%s",
            doc.title, package_id, doc.document_type.value,
        )
        return doc

    async def check_completeness(
        self, package_id: str,
    ) -> CompletenessCheck:
        """Check evidence completeness against ISO 14064-3 requirements."""
        package = self._packages.get(package_id)
        if not package:
            raise ValueError(f"Package {package_id} not found")

        total = 0
        met = 0
        missing = 0
        incomplete = 0
        optional = 0
        gaps: List[Dict[str, str]] = []

        for wp in package.workpapers:
            if not wp.required:
                optional += 1
                continue

            total += 1
            if wp.status == EvidenceStatus.PROVIDED or wp.documents:
                met += 1
            elif wp.status == EvidenceStatus.INCOMPLETE:
                incomplete += 1
                gaps.append({
                    "requirement_id": wp.requirement_id,
                    "name": wp.name,
                    "status": "incomplete",
                    "action": f"Complete documentation for {wp.name}",
                })
            else:
                missing += 1
                gaps.append({
                    "requirement_id": wp.requirement_id,
                    "name": wp.name,
                    "status": "missing",
                    "action": f"Upload {wp.name} documentation",
                })

        completeness_pct = (met / max(total, 1)) * 100.0
        ready = missing == 0 and incomplete == 0

        check = CompletenessCheck(
            total_requirements=total,
            requirements_met=met,
            requirements_missing=missing,
            requirements_incomplete=incomplete,
            requirements_optional=optional,
            completeness_pct=round(completeness_pct, 2),
            ready_for_submission=ready,
            gaps=gaps,
        )

        if self.config.enable_provenance:
            check.provenance_hash = _compute_hash(check)

        package.completeness = check
        if ready:
            package.status = WorkflowStatus.READY_FOR_SUBMISSION

        self.logger.info(
            "Completeness check: %d/%d requirements met (%.1f%%), "
            "ready=%s, gaps=%d",
            met, total, completeness_pct, ready, len(gaps),
        )
        return check

    async def submit_assurance_request(
        self,
        package_id: str,
        scope_of_engagement: Optional[List[str]] = None,
        fee_usd: float = 0.0,
    ) -> AssuranceRequest:
        """Submit assurance engagement request to provider."""
        package = self._packages.get(package_id)
        if not package:
            raise ValueError(f"Package {package_id} not found")

        if package.completeness and not package.completeness.ready_for_submission:
            self.logger.warning(
                "Submitting package %s that is not fully complete "
                "(%.1f%% completeness)",
                package_id, package.completeness.completeness_pct,
            )

        scope = scope_of_engagement or [
            "Scope 1 GHG emissions",
            "Scope 2 GHG emissions (location and market-based)",
            "Scope 3 GHG emissions (all 15 categories)",
            "Interim target trajectory and linearity",
            "Annual carbon budget performance",
            "SBTi target alignment",
        ]

        request = AssuranceRequest(
            organization_name=self.config.organization_name,
            reporting_year=self.config.reporting_year,
            assurance_level=self.config.assurance_level,
            assurance_standard=self.config.assurance_standard,
            provider=self.config.provider,
            scope_of_engagement=scope,
            status=WorkflowStatus.SUBMITTED,
            evidence_package_id=package_id,
            submitted_at=utcnow(),
            estimated_completion_weeks=8 if self.config.assurance_level == AssuranceLevel.LIMITED else 12,
            fee_usd=fee_usd,
        )

        if self.config.enable_provenance:
            request.provenance_hash = _compute_hash(request)

        self._requests[request.request_id] = request
        package.status = WorkflowStatus.SUBMITTED

        # Attempt API submission
        if self.config.provider_api_key and self.config.provider_api_url:
            api_result = await self._submit_to_provider_api(request, package)
            if api_result.get("success"):
                request.status = WorkflowStatus.UNDER_REVIEW

        self.logger.info(
            "Assurance request submitted: %s, level=%s, provider=%s, "
            "scope=%d items",
            request.request_id, request.assurance_level.value,
            request.provider.value, len(scope),
        )
        return request

    async def update_workflow_status(
        self,
        request_id: str,
        new_status: WorkflowStatus,
        notes: str = "",
    ) -> Optional[AssuranceRequest]:
        """Update assurance workflow status."""
        request = self._requests.get(request_id)
        if not request:
            return None

        request.status = new_status
        self.logger.info(
            "Workflow updated: %s -> %s (%s)",
            request_id, new_status.value, notes,
        )
        return request

    async def get_evidence_summary(
        self, package_id: str,
    ) -> Dict[str, Any]:
        """Get evidence package summary."""
        package = self._packages.get(package_id)
        if not package:
            return {"error": f"Package {package_id} not found"}

        by_category: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        for wp in package.workpapers:
            by_category[wp.category] = by_category.get(wp.category, 0) + 1
            by_status[wp.status.value] = by_status.get(wp.status.value, 0) + 1

        return {
            "package_id": package.package_id,
            "organization": package.organization_name,
            "reporting_year": package.reporting_year,
            "assurance_level": package.assurance_level.value,
            "standard": package.assurance_standard.value,
            "workpapers_total": len(package.workpapers),
            "documents_total": len(package.documents),
            "by_category": by_category,
            "by_status": by_status,
            "version": package.version,
            "status": package.status.value,
            "completeness_pct": package.completeness.completeness_pct if package.completeness else 0.0,
            "ready_for_submission": package.completeness.ready_for_submission if package.completeness else False,
        }

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "organization": self.config.organization_name,
            "assurance_level": self.config.assurance_level.value,
            "standard": self.config.assurance_standard.value,
            "provider": self.config.provider.value,
            "packages_created": len(self._packages),
            "requests_submitted": len(self._requests),
            "api_configured": bool(self.config.provider_api_key),
        }

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # -------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------

    async def _submit_to_provider_api(
        self,
        request: AssuranceRequest,
        package: EvidencePackage,
    ) -> Dict[str, Any]:
        """Submit to provider API."""
        try:
            import httpx

            if not self._http_client:
                self._http_client = httpx.AsyncClient(
                    base_url=self.config.provider_api_url,
                    headers={"Authorization": f"Bearer {self.config.provider_api_key}"},
                    timeout=60.0,
                )

            payload = {
                "organization": request.organization_name,
                "reporting_year": request.reporting_year,
                "assurance_level": request.assurance_level.value,
                "standard": request.assurance_standard.value,
                "scope": request.scope_of_engagement,
                "documents_count": len(package.documents),
                "completeness_pct": package.completeness.completeness_pct if package.completeness else 0,
            }

            response = await self._http_client.post("/engagements", json=payload)
            if response.status_code in (200, 201):
                return {"success": True, "engagement_id": response.json().get("engagement_id", "")}
            else:
                return {"success": False, "error": response.text}

        except ImportError:
            return {"success": False, "reason": "httpx not available"}
        except Exception as exc:
            self.logger.error("Provider API submission failed: %s", exc)
            return {"success": False, "reason": str(exc)}
