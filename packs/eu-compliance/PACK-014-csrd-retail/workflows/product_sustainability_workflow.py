# -*- coding: utf-8 -*-
"""
Product Sustainability Workflow
====================================

4-phase workflow for product-level sustainability assessment within
PACK-014 CSRD Retail and Consumer Goods Pack.

Phases:
    1. ProductCatalog     -- Inventory products requiring DPP/PEF
    2. DPPGeneration      -- Generate Digital Product Passport data
    3. GreenClaimsAudit   -- Check claims against ECGT prohibitions
    4. ComplianceReport   -- Consolidated product sustainability report

Author: GreenLang Team
Version: 14.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
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


class ProductCategory(str, Enum):
    """Product categories for DPP/PEF requirements."""
    TEXTILES = "textiles"
    ELECTRONICS = "electronics"
    BATTERIES = "batteries"
    FURNITURE = "furniture"
    FOOD = "food"
    COSMETICS = "cosmetics"
    DETERGENTS = "detergents"
    TOYS = "toys"
    GENERAL = "general"


class ClaimStatus(str, Enum):
    """Green claim audit status."""
    SUBSTANTIATED = "substantiated"
    UNSUBSTANTIATED = "unsubstantiated"
    PROHIBITED = "prohibited"
    NEEDS_REVIEW = "needs_review"


class DPPStatus(str, Enum):
    """Digital Product Passport readiness status."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_STARTED = "not_started"
    NOT_REQUIRED = "not_required"


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


class ProductRecord(BaseModel):
    """Product requiring sustainability assessment."""
    product_id: str = Field(default_factory=lambda: f"prod-{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="Product name")
    category: ProductCategory = Field(default=ProductCategory.GENERAL)
    sku: str = Field(default="")
    brand: str = Field(default="")
    materials: List[str] = Field(default_factory=list)
    weight_kg: float = Field(default=0.0, ge=0.0)
    country_of_origin: str = Field(default="")
    supplier_id: str = Field(default="")
    requires_dpp: bool = Field(default=False)
    carbon_footprint_kgco2e: Optional[float] = Field(None, ge=0.0)
    recyclable_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    recycled_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    durability_years: Optional[float] = Field(None, ge=0.0)
    repairability_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    energy_label: str = Field(default="", description="EU energy label class")


class GreenClaim(BaseModel):
    """A green/environmental claim made about a product."""
    claim_id: str = Field(default_factory=lambda: f"clm-{uuid.uuid4().hex[:6]}")
    product_id: str = Field(default="")
    claim_text: str = Field(default="", description="The claim text")
    claim_type: str = Field(default="", description="carbon_neutral|eco_friendly|sustainable|biodegradable|etc")
    evidence_available: bool = Field(default=False)
    third_party_verified: bool = Field(default=False)
    certification_name: str = Field(default="")
    scope_specified: bool = Field(default=False, description="Does claim specify scope?")


class DPPRecord(BaseModel):
    """Digital Product Passport record for a product."""
    product_id: str = Field(default="")
    product_name: str = Field(default="")
    dpp_status: DPPStatus = Field(default=DPPStatus.NOT_STARTED)
    fields_complete: int = Field(default=0, ge=0)
    fields_required: int = Field(default=0, ge=0)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    missing_fields: List[str] = Field(default_factory=list)
    qr_code_ready: bool = Field(default=False)


class ClaimAuditResult(BaseModel):
    """Result of auditing a green claim."""
    claim_id: str = Field(default="")
    product_id: str = Field(default="")
    claim_text: str = Field(default="")
    status: ClaimStatus = Field(default=ClaimStatus.NEEDS_REVIEW)
    reason: str = Field(default="")
    ecgt_article: str = Field(default="", description="Relevant ECGT article")
    recommendation: str = Field(default="")


class PEFResult(BaseModel):
    """Product Environmental Footprint result."""
    product_id: str = Field(default="")
    product_name: str = Field(default="")
    pef_score: float = Field(default=0.0, ge=0.0)
    climate_change_kgco2e: float = Field(default=0.0, ge=0.0)
    resource_use_mj: float = Field(default=0.0, ge=0.0)
    water_use_m3: float = Field(default=0.0, ge=0.0)
    pef_category: str = Field(default="")
    benchmark_percentile: float = Field(default=0.0, ge=0.0, le=100.0)


class ProductSustainabilityInput(BaseModel):
    """Input data model for ProductSustainabilityWorkflow."""
    products: List[ProductRecord] = Field(default_factory=list)
    claims: List[GreenClaim] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class ProductSustainabilityResult(BaseModel):
    """Complete result from product sustainability workflow."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="product_sustainability")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    dpp_coverage: Dict[str, Any] = Field(default_factory=dict)
    dpp_records: List[DPPRecord] = Field(default_factory=list)
    claims_audit: List[ClaimAuditResult] = Field(default_factory=list)
    pef_results: List[PEFResult] = Field(default_factory=list)
    compliance: Dict[str, Any] = Field(default_factory=dict)
    total_products: int = Field(default=0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# ECGT PROHIBITED CLAIMS
# =============================================================================

ECGT_PROHIBITED_TERMS: List[str] = [
    "carbon neutral",
    "climate neutral",
    "carbon positive",
    "climate positive",
    "co2 neutral",
    "carbon free",
    "climate compensated",
    "carbon compensated",
    "reduced carbon footprint",  # unless substantiated with full LCA
    "environmentally friendly",
    "eco-friendly",
    "green",
    "nature's friend",
    "ecologically safe",
    "climate friendly",
    "gentle on the environment",
    "not harmful to the environment",
]

# DPP required fields per category
DPP_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "textiles": [
        "material_composition", "country_of_manufacturing", "recyclability",
        "recycled_content", "carbon_footprint", "water_consumption",
        "microplastic_release", "durability", "care_instructions",
        "supplier_chain_info",
    ],
    "electronics": [
        "material_composition", "energy_efficiency", "repairability_score",
        "spare_parts_availability", "recycled_content", "carbon_footprint",
        "expected_lifetime", "battery_info", "hazardous_substances",
    ],
    "batteries": [
        "chemistry", "capacity", "recycled_content", "carbon_footprint",
        "expected_cycles", "collection_info", "hazardous_substances",
        "performance_parameters",
    ],
    "general": [
        "material_composition", "country_of_origin", "recyclability",
        "recycled_content", "carbon_footprint",
    ],
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ProductSustainabilityWorkflow:
    """
    4-phase product sustainability workflow.

    Inventories products, generates Digital Product Passport data,
    audits green claims against ECGT requirements, and produces
    consolidated compliance reports.

    Example:
        >>> wf = ProductSustainabilityWorkflow()
        >>> inp = ProductSustainabilityInput(products=[...], claims=[...])
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ProductSustainabilityWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._dpp_records: List[DPPRecord] = []
        self._claim_results: List[ClaimAuditResult] = []
        self._pef_results: List[PEFResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[ProductSustainabilityInput] = None,
        products: Optional[List[ProductRecord]] = None,
        claims: Optional[List[GreenClaim]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ProductSustainabilityResult:
        """Execute the 4-phase product sustainability workflow."""
        if input_data is None:
            input_data = ProductSustainabilityInput(
                products=products or [], claims=claims or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting product sustainability workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_product_catalog(input_data))
            phase_results.append(await self._phase_dpp_generation(input_data))
            phase_results.append(await self._phase_green_claims_audit(input_data))
            phase_results.append(await self._phase_compliance_report(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Product sustainability workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        dpp_complete = sum(1 for d in self._dpp_records if d.dpp_status == DPPStatus.COMPLETE)
        dpp_required = sum(1 for d in self._dpp_records if d.dpp_status != DPPStatus.NOT_REQUIRED)
        claims_ok = sum(1 for c in self._claim_results if c.status == ClaimStatus.SUBSTANTIATED)
        claims_prohibited = sum(1 for c in self._claim_results if c.status == ClaimStatus.PROHIBITED)

        result = ProductSustainabilityResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            dpp_coverage={
                "total_products": len(input_data.products),
                "dpp_required": dpp_required,
                "dpp_complete": dpp_complete,
                "coverage_pct": round(dpp_complete / max(dpp_required, 1) * 100, 2),
            },
            dpp_records=self._dpp_records,
            claims_audit=self._claim_results,
            pef_results=self._pef_results,
            compliance={
                "dpp_compliance_pct": round(dpp_complete / max(dpp_required, 1) * 100, 2),
                "claims_substantiated": claims_ok,
                "claims_prohibited": claims_prohibited,
                "claims_total": len(self._claim_results),
            },
            total_products=len(input_data.products),
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Product Catalog
    # -------------------------------------------------------------------------

    async def _phase_product_catalog(self, input_data: ProductSustainabilityInput) -> PhaseResult:
        """Inventory products requiring DPP/PEF assessment."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        category_counts: Dict[str, int] = {}
        dpp_required_count = 0

        for prod in input_data.products:
            category_counts[prod.category.value] = category_counts.get(prod.category.value, 0) + 1
            if prod.requires_dpp or prod.category in (ProductCategory.TEXTILES, ProductCategory.ELECTRONICS, ProductCategory.BATTERIES):
                dpp_required_count += 1
            if not prod.materials:
                warnings.append(f"Product {prod.product_id}: no materials specified")

        outputs["total_products"] = len(input_data.products)
        outputs["category_distribution"] = category_counts
        outputs["dpp_required_count"] = dpp_required_count
        outputs["total_claims"] = len(input_data.claims)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 1 ProductCatalog: %d products, %d require DPP", len(input_data.products), dpp_required_count)
        return PhaseResult(
            phase_name="product_catalog", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: DPP Generation
    # -------------------------------------------------------------------------

    async def _phase_dpp_generation(self, input_data: ProductSustainabilityInput) -> PhaseResult:
        """Generate Digital Product Passport data for applicable products."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._dpp_records = []
        self._pef_results = []

        for prod in input_data.products:
            needs_dpp = prod.requires_dpp or prod.category in (
                ProductCategory.TEXTILES, ProductCategory.ELECTRONICS, ProductCategory.BATTERIES
            )
            if not needs_dpp:
                self._dpp_records.append(DPPRecord(
                    product_id=prod.product_id, product_name=prod.name,
                    dpp_status=DPPStatus.NOT_REQUIRED,
                ))
                continue

            required_fields = DPP_REQUIRED_FIELDS.get(prod.category.value, DPP_REQUIRED_FIELDS["general"])
            available_fields = self._check_available_fields(prod, required_fields)
            complete_count = sum(1 for v in available_fields.values() if v)
            total_required = len(required_fields)
            missing = [f for f, v in available_fields.items() if not v]
            completeness = (complete_count / max(total_required, 1)) * 100

            if completeness >= 90:
                status = DPPStatus.COMPLETE
            elif completeness >= 50:
                status = DPPStatus.PARTIAL
            else:
                status = DPPStatus.NOT_STARTED

            self._dpp_records.append(DPPRecord(
                product_id=prod.product_id, product_name=prod.name,
                dpp_status=status, fields_complete=complete_count,
                fields_required=total_required,
                completeness_pct=round(completeness, 2),
                missing_fields=missing,
                qr_code_ready=status == DPPStatus.COMPLETE,
            ))

            # PEF result
            if prod.carbon_footprint_kgco2e is not None:
                self._pef_results.append(PEFResult(
                    product_id=prod.product_id, product_name=prod.name,
                    climate_change_kgco2e=prod.carbon_footprint_kgco2e,
                    pef_category=prod.category.value,
                ))

        outputs["dpp_generated"] = sum(1 for d in self._dpp_records if d.dpp_status != DPPStatus.NOT_REQUIRED)
        outputs["dpp_complete"] = sum(1 for d in self._dpp_records if d.dpp_status == DPPStatus.COMPLETE)
        outputs["pef_calculated"] = len(self._pef_results)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 2 DPPGeneration: %d DPPs generated, %d complete", outputs["dpp_generated"], outputs["dpp_complete"])
        return PhaseResult(
            phase_name="dpp_generation", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    def _check_available_fields(self, prod: ProductRecord, required: List[str]) -> Dict[str, bool]:
        """Check which DPP fields are available for a product."""
        checks: Dict[str, bool] = {}
        for field in required:
            if field == "material_composition":
                checks[field] = bool(prod.materials)
            elif field == "country_of_origin" or field == "country_of_manufacturing":
                checks[field] = bool(prod.country_of_origin)
            elif field == "recyclability":
                checks[field] = prod.recyclable_pct > 0
            elif field == "recycled_content":
                checks[field] = prod.recycled_content_pct > 0
            elif field == "carbon_footprint":
                checks[field] = prod.carbon_footprint_kgco2e is not None
            elif field == "durability":
                checks[field] = prod.durability_years is not None
            elif field == "repairability_score":
                checks[field] = prod.repairability_score is not None
            elif field == "energy_efficiency":
                checks[field] = bool(prod.energy_label)
            else:
                checks[field] = False  # Cannot verify without additional data
        return checks

    # -------------------------------------------------------------------------
    # Phase 3: Green Claims Audit
    # -------------------------------------------------------------------------

    async def _phase_green_claims_audit(self, input_data: ProductSustainabilityInput) -> PhaseResult:
        """Check claims against ECGT prohibitions."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._claim_results = []

        for claim in input_data.claims:
            result = self._audit_claim(claim)
            self._claim_results.append(result)

        substantiated = sum(1 for c in self._claim_results if c.status == ClaimStatus.SUBSTANTIATED)
        prohibited = sum(1 for c in self._claim_results if c.status == ClaimStatus.PROHIBITED)
        unsubstantiated = sum(1 for c in self._claim_results if c.status == ClaimStatus.UNSUBSTANTIATED)

        outputs["total_claims"] = len(self._claim_results)
        outputs["substantiated"] = substantiated
        outputs["prohibited"] = prohibited
        outputs["unsubstantiated"] = unsubstantiated
        outputs["needs_review"] = sum(1 for c in self._claim_results if c.status == ClaimStatus.NEEDS_REVIEW)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 3 GreenClaimsAudit: %d prohibited, %d substantiated", prohibited, substantiated)
        return PhaseResult(
            phase_name="green_claims_audit", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    def _audit_claim(self, claim: GreenClaim) -> ClaimAuditResult:
        """Audit a single green claim against ECGT requirements."""
        claim_lower = claim.claim_text.lower()

        # Check prohibited terms
        for term in ECGT_PROHIBITED_TERMS:
            if term in claim_lower:
                return ClaimAuditResult(
                    claim_id=claim.claim_id, product_id=claim.product_id,
                    claim_text=claim.claim_text,
                    status=ClaimStatus.PROHIBITED,
                    reason=f"Contains prohibited term: '{term}'",
                    ecgt_article="Article 6(2)",
                    recommendation=f"Remove or rephrase claim. '{term}' is prohibited under ECGT.",
                )

        # Check substantiation
        if claim.evidence_available and claim.third_party_verified and claim.scope_specified:
            return ClaimAuditResult(
                claim_id=claim.claim_id, product_id=claim.product_id,
                claim_text=claim.claim_text,
                status=ClaimStatus.SUBSTANTIATED,
                reason="Claim is supported by evidence, third-party verified, and scope is specified",
                ecgt_article="Article 3",
            )

        if claim.evidence_available and not claim.third_party_verified:
            return ClaimAuditResult(
                claim_id=claim.claim_id, product_id=claim.product_id,
                claim_text=claim.claim_text,
                status=ClaimStatus.NEEDS_REVIEW,
                reason="Evidence available but not third-party verified",
                ecgt_article="Article 5",
                recommendation="Obtain third-party verification for the claim",
            )

        return ClaimAuditResult(
            claim_id=claim.claim_id, product_id=claim.product_id,
            claim_text=claim.claim_text,
            status=ClaimStatus.UNSUBSTANTIATED,
            reason="Insufficient evidence to support claim",
            ecgt_article="Article 3",
            recommendation="Gather scientific evidence and obtain verification",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Compliance Report
    # -------------------------------------------------------------------------

    async def _phase_compliance_report(self, input_data: ProductSustainabilityInput) -> PhaseResult:
        """Generate consolidated compliance report."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}

        dpp_complete = sum(1 for d in self._dpp_records if d.dpp_status == DPPStatus.COMPLETE)
        dpp_required = sum(1 for d in self._dpp_records if d.dpp_status != DPPStatus.NOT_REQUIRED)
        claims_ok = sum(1 for c in self._claim_results if c.status == ClaimStatus.SUBSTANTIATED)
        claims_total = len(self._claim_results)

        outputs["dpp_compliance_pct"] = round(dpp_complete / max(dpp_required, 1) * 100, 2)
        outputs["claims_compliance_pct"] = round(claims_ok / max(claims_total, 1) * 100, 2)
        outputs["overall_product_count"] = len(input_data.products)
        outputs["action_items_count"] = (
            sum(1 for d in self._dpp_records if d.dpp_status == DPPStatus.PARTIAL or d.dpp_status == DPPStatus.NOT_STARTED)
            + sum(1 for c in self._claim_results if c.status in (ClaimStatus.PROHIBITED, ClaimStatus.UNSUBSTANTIATED))
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 4 ComplianceReport: DPP=%.1f%%, Claims=%.1f%%", outputs["dpp_compliance_pct"], outputs["claims_compliance_pct"])
        return PhaseResult(
            phase_name="compliance_report", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ProductSustainabilityResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
