# -*- coding: utf-8 -*-
"""
DDS Generation Workflow
========================

Six-phase Due Diligence Statement (DDS) generation workflow -- the primary
EUDR compliance workflow. Orchestrates the full lifecycle from supplier
onboarding through to EU Information System submission.

Regulatory Context:
    Per EU Regulation 2023/1115 (EUDR - Deforestation-Free Products Regulation):
    - Article 4: Operators must exercise due diligence before placing relevant
      commodities/products on the EU market or exporting them
    - Article 9: Due Diligence Statements (DDS) must be submitted to the EU
      Information System before placing goods on the market
    - Article 10: DDS must contain information as specified in Annex II,
      including geolocation of all plots of land, risk assessment results,
      and risk mitigation measures
    - Article 29: Country benchmarking determines standard vs simplified DD

    The seven relevant commodities are: cattle, cocoa, coffee, oil palm,
    rubber, soya, and wood (plus derived products per Annex I).

Phases:
    1. Supplier onboarding - Collect profiles, validate EORI, import geolocation
    2. Geolocation collection - Validate coordinates/polygons, check area rules
    3. Document collection - Authenticate certificates, verify permits
    4. Risk assessment - Country/supplier/commodity/document risk scoring
    5. DDS generation - Assemble DDS per Annex II, attach all evidence
    6. Review and submit - Human review, DDS validation, EU IS submission

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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


class DDType(str, Enum):
    """Due diligence type based on country risk benchmarking."""
    STANDARD = "standard"
    SIMPLIFIED = "simplified"


class RiskLevel(str, Enum):
    """Risk classification level."""
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class EUDRCommodity(str, Enum):
    """EUDR-relevant commodities per Article 1."""
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class CertificationType(str, Enum):
    """Supported sustainability certification schemes."""
    FSC = "FSC"
    PEFC = "PEFC"
    RSPO = "RSPO"
    ISCC = "ISCC"
    RAINFOREST_ALLIANCE = "RA"
    UTZ = "UTZ"


class ReviewDecision(str, Enum):
    """Human review decision options."""
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUIRED = "revision_required"


# EUDR cutoff date: December 31, 2020 (Article 2(12))
EUDR_CUTOFF_DATE = "2020-12-31"

# Maximum area threshold for point-only geolocation (Article 9(1)(d))
POINT_ONLY_MAX_AREA_HA = 4.0

# Coordinate precision requirement: 6 decimal places (WGS84)
MIN_COORDINATE_DECIMALS = 6

# Country risk benchmarking categories (Article 29)
HIGH_RISK_COUNTRIES = {
    "BR", "CD", "CM", "CO", "CI", "EC", "GA", "GH", "GT", "GN",
    "HN", "ID", "KH", "LA", "LR", "MG", "MM", "MY", "MZ", "NG",
    "PA", "PE", "PG", "PH", "SL", "TZ", "TH", "UG", "VN",
}

LOW_RISK_COUNTRIES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "NO", "IS", "CH", "LI", "GB", "AU", "NZ", "JP", "KR", "CA",
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    checkpoints: Dict[str, Any] = Field(default_factory=dict)
    state: Dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[datetime] = Field(None)
    last_checkpoint_at: Optional[datetime] = Field(None)

    class Config:
        arbitrary_types_allowed = True


class SupplierData(BaseModel):
    """Supplier profile data for onboarding."""
    supplier_id: Optional[str] = Field(None, description="Existing supplier ID")
    supplier_name: str = Field(..., min_length=1, description="Legal entity name")
    country_code: str = Field(..., min_length=2, max_length=2, description="ISO 3166 alpha-2")
    eori_number: Optional[str] = Field(None, description="EORI number for EU-based operators")
    contact_email: str = Field(..., description="Primary contact email")
    contact_name: Optional[str] = Field(None, description="Contact person")
    commodities: List[EUDRCommodity] = Field(default_factory=list, description="Traded commodities")
    certifications: List[str] = Field(default_factory=list, description="Held certifications")
    plot_count: int = Field(default=0, ge=0, description="Number of production plots")

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase alpha-2."""
        if not v.isalpha() or not v.isupper() or len(v) != 2:
            raise ValueError(f"Country code must be uppercase ISO 3166 alpha-2, got: {v}")
        return v


class GeolocationPoint(BaseModel):
    """A single geolocation point (WGS84)."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")

    @field_validator("latitude", "longitude")
    @classmethod
    def validate_precision(cls, v: float) -> float:
        """Ensure coordinate has sufficient decimal precision."""
        return round(v, 6)


class PlotGeolocation(BaseModel):
    """Geolocation data for a production plot."""
    plot_id: str = Field(..., description="Unique plot identifier")
    supplier_id: str = Field(..., description="Owning supplier")
    country_code: str = Field(..., min_length=2, max_length=2)
    centroid: Optional[GeolocationPoint] = Field(None, description="Plot centroid point")
    polygon_points: List[GeolocationPoint] = Field(
        default_factory=list, description="Polygon boundary points for >= 4ha plots"
    )
    area_hectares: float = Field(default=0.0, ge=0.0, description="Plot area in hectares")
    commodity: Optional[EUDRCommodity] = Field(None, description="Primary commodity grown")
    production_start_date: Optional[str] = Field(None, description="YYYY-MM-DD")


class CertificationRecord(BaseModel):
    """Sustainability certification record."""
    cert_id: str = Field(..., description="Certificate number")
    cert_type: CertificationType = Field(..., description="Certification scheme")
    supplier_id: str = Field(..., description="Certificate holder supplier ID")
    issue_date: str = Field(..., description="Issue date YYYY-MM-DD")
    expiry_date: str = Field(..., description="Expiry date YYYY-MM-DD")
    scope: List[str] = Field(default_factory=list, description="Certified commodities/products")
    verified: bool = Field(default=False, description="Whether cert has been verified")
    document_hash: Optional[str] = Field(None, description="SHA-256 of certificate document")


class RiskScore(BaseModel):
    """Composite risk assessment score."""
    country_risk: float = Field(default=0.0, ge=0.0, le=100.0)
    supplier_risk: float = Field(default=0.0, ge=0.0, le=100.0)
    commodity_risk: float = Field(default=0.0, ge=0.0, le=100.0)
    document_risk: float = Field(default=0.0, ge=0.0, le=100.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_level: RiskLevel = Field(default=RiskLevel.STANDARD)
    dd_type: DDType = Field(default=DDType.STANDARD)
    scoring_weights: Dict[str, float] = Field(default_factory=dict)


class DDSContent(BaseModel):
    """Due Diligence Statement content per Annex II."""
    dds_id: str = Field(..., description="DDS unique identifier")
    reference_number: Optional[str] = Field(None, description="EU IS reference number")
    operator_name: str = Field(default="", description="Operator legal name")
    operator_eori: Optional[str] = Field(None, description="Operator EORI")
    commodities: List[str] = Field(default_factory=list, description="Commodities covered")
    suppliers: List[Dict[str, Any]] = Field(default_factory=list, description="Supplier details")
    geolocations: List[Dict[str, Any]] = Field(default_factory=list, description="Plot geolocations")
    risk_assessment: Dict[str, Any] = Field(default_factory=dict, description="Risk summary")
    mitigation_measures: List[str] = Field(default_factory=list)
    certifications: List[Dict[str, Any]] = Field(default_factory=list)
    dd_type: DDType = Field(default=DDType.STANDARD)
    declaration_text: str = Field(default="", description="Compliance declaration")
    generated_at: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")


class DDSGenerationInput(BaseModel):
    """Input data for the DDS generation workflow."""
    operator_name: str = Field(..., description="Name of the operator/importer")
    operator_eori: Optional[str] = Field(None, description="EORI number of operator")
    suppliers: List[SupplierData] = Field(..., min_length=1, description="Suppliers to include")
    geolocations: List[PlotGeolocation] = Field(default_factory=list, description="Plot data")
    certifications: List[CertificationRecord] = Field(default_factory=list, description="Certs")
    commodities: List[EUDRCommodity] = Field(default_factory=list, description="Target commodities")
    auto_submit: bool = Field(default=False, description="Auto-submit for low-risk DDS")
    config: Dict[str, Any] = Field(default_factory=dict)


class DDSGenerationResult(BaseModel):
    """Complete result from the DDS generation workflow."""
    workflow_name: str = Field(default="dds_generation")
    status: PhaseStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    dds_id: str = Field(..., description="Generated DDS identifier")
    reference_number: Optional[str] = Field(None, description="EU IS reference number")
    dd_type: DDType = Field(default=DDType.STANDARD)
    risk_level: RiskLevel = Field(default=RiskLevel.STANDARD)
    composite_risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    suppliers_processed: int = Field(default=0, ge=0)
    plots_validated: int = Field(default=0, ge=0)
    certifications_verified: int = Field(default=0, ge=0)
    dds_content: Optional[DDSContent] = Field(None)
    submitted: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    execution_id: str = Field(default="")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)


# =============================================================================
# DDS GENERATION WORKFLOW
# =============================================================================


class DDSGenerationWorkflow:
    """
    Six-phase Due Diligence Statement generation workflow.

    Orchestrates the complete EUDR DDS lifecycle from supplier onboarding
    through EU Information System submission. This is the primary compliance
    workflow for organizations placing relevant commodities on the EU market.

    Phase durations (typical):
        1. Supplier Onboarding:      ~5 business days
        2. Geolocation Collection:   ~3 business days
        3. Document Collection:      ~3 business days
        4. Risk Assessment:          ~2 business days
        5. DDS Generation:           ~2 business days
        6. Review and Submit:        ~2 business days

    Agent Dependencies:
        - EUDR-001 (Supply Chain Mapper)
        - EUDR-002 (Geolocation Verification)
        - EUDR-006 (Plot Boundary)
        - EUDR-007 (GPS Validator)
        - EUDR-008 (Multi-Tier Supplier)
        - EUDR-012 (Document Authentication)
        - EUDR-016 (Country Risk Classifier)
        - EUDR-017 (Supplier Risk Scorer)
        - EUDR-018 (Commodity Risk Analyzer)
        - EUDR-028 (Risk Assessment Coordinator)
        - EUDR-030 (Documentation Generator)
        - EUDR-036 (EU IS Interface)
        - EUDR-037 (DDS Creator)
        - EUDR-038 (Reference Number Generator)
        - DATA-001 (PDF Extractor)
        - DATA-002 (Excel/CSV Normalizer)
        - FOUND-005 (Citations & Evidence)

    Attributes:
        config: Workflow configuration dict.
        logger: Logger instance.
        _execution_id: Unique execution identifier.
        _phase_results: Accumulated phase results.
        _checkpoint_store: Checkpoint data for resume support.

    Example:
        >>> wf = DDSGenerationWorkflow()
        >>> result = await wf.run(DDSGenerationInput(
        ...     operator_name="Acme Imports GmbH",
        ...     operator_eori="DE123456789012345",
        ...     suppliers=[SupplierData(
        ...         supplier_name="Brazil Coffee Co",
        ...         country_code="BR",
        ...         contact_email="info@brazcoffee.com",
        ...         commodities=[EUDRCommodity.COFFEE],
        ...     )],
        ...     commodities=[EUDRCommodity.COFFEE],
        ... ))
        >>> assert result.status == PhaseStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the DDSGenerationWorkflow.

        Args:
            config: Optional configuration dict with keys like
                'operator_id', 'auto_submit_threshold', 'strict_mode'.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.DDSGenerationWorkflow")
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []
        self._checkpoint_store: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(self, input_data: DDSGenerationInput) -> DDSGenerationResult:
        """
        Execute the full 6-phase DDS generation workflow.

        Args:
            input_data: Validated input containing operator info, suppliers,
                geolocations, certifications, and configuration.

        Returns:
            DDSGenerationResult with DDS content, risk scores, and submission status.
        """
        started_at = datetime.utcnow()
        dds_id = f"DDS-{self._execution_id[:12]}"

        self.logger.info(
            "Starting DDS generation workflow execution_id=%s operator=%s",
            self._execution_id, input_data.operator_name,
        )

        context = WorkflowContext(
            execution_id=self._execution_id,
            config={**self.config, **input_data.config},
            started_at=started_at,
            state={
                "input_data": input_data.model_dump(),
                "dds_id": dds_id,
                "operator_name": input_data.operator_name,
                "operator_eori": input_data.operator_eori,
                "auto_submit": input_data.auto_submit,
            },
        )

        phase_handlers: List[Tuple[str, Any]] = [
            ("supplier_onboarding", self._phase_1_supplier_onboarding),
            ("geolocation_collection", self._phase_2_geolocation_collection),
            ("document_collection", self._phase_3_document_collection),
            ("risk_assessment", self._phase_4_risk_assessment),
            ("dds_generation", self._phase_5_dds_generation),
            ("review_and_submit", self._phase_6_review_and_submit),
        ]

        overall_status = PhaseStatus.COMPLETED
        resume_from = self._get_resume_phase(context)

        for phase_name, handler in phase_handlers:
            # Skip already-completed phases on resume
            if resume_from and phase_name != resume_from:
                completed = self._load_checkpoint(context, phase_name)
                if completed:
                    self._phase_results.append(completed)
                    continue

            if resume_from and phase_name == resume_from:
                resume_from = None  # Start executing from this phase

            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase_name)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (
                    datetime.utcnow() - phase_start
                ).total_seconds()
            except Exception as exc:
                self.logger.error(
                    "Phase '%s' failed: %s", phase_name, exc, exc_info=True,
                )
                phase_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    outputs={"error": str(exc)},
                    provenance_hash=self._hash({"error": str(exc)}),
                )

            self._phase_results.append(phase_result)
            context.phase_results = list(self._phase_results)
            self._save_checkpoint(context, phase_name, phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                # Phases 1-4 are critical; halt on failure
                if phase_name in (
                    "supplier_onboarding", "geolocation_collection",
                    "document_collection", "risk_assessment",
                ):
                    self.logger.error(
                        "Critical phase '%s' failed; halting workflow.", phase_name,
                    )
                    break

        completed_at = datetime.utcnow()

        # Extract final state
        risk_score = context.state.get("risk_score", {})
        dds_content = context.state.get("dds_content")

        provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
            "dds_id": dds_id,
        })

        self.logger.info(
            "DDS generation workflow finished execution_id=%s status=%s dds_id=%s "
            "duration=%.2fs",
            self._execution_id, overall_status.value, dds_id,
            (completed_at - started_at).total_seconds(),
        )

        return DDSGenerationResult(
            status=overall_status,
            phases=self._phase_results,
            dds_id=dds_id,
            reference_number=context.state.get("reference_number"),
            dd_type=DDType(risk_score.get("dd_type", "standard")),
            risk_level=RiskLevel(risk_score.get("risk_level", "standard")),
            composite_risk_score=risk_score.get("composite_score", 0.0),
            suppliers_processed=context.state.get("suppliers_processed", 0),
            plots_validated=context.state.get("plots_validated", 0),
            certifications_verified=context.state.get("certifications_verified", 0),
            dds_content=DDSContent(**dds_content) if isinstance(dds_content, dict) else dds_content,
            submitted=context.state.get("submitted", False),
            provenance_hash=provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Supplier Onboarding (~5 business days typical)
    # -------------------------------------------------------------------------

    async def _phase_1_supplier_onboarding(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Collect supplier profiles, validate EORI/company data, import
        geolocation data, and gather certifications.

        Uses:
            - EUDR-001 (Supply Chain Mapper)
            - EUDR-008 (Multi-Tier Supplier)
            - DATA-001 (PDF Extractor)
            - DATA-002 (Excel/CSV Normalizer)

        Steps:
            1. Parse and normalize supplier profiles from input
            2. Validate EORI numbers for EU-based operators
            3. Check for duplicate suppliers in the system
            4. Validate commodity classifications against EUDR Annex I
            5. Assign supplier IDs and create registry entries
        """
        phase_name = "supplier_onboarding"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        input_data = context.state.get("input_data", {})
        suppliers_raw = input_data.get("suppliers", [])

        if not suppliers_raw:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs={"error": "No suppliers provided"},
                provenance_hash=self._hash({"phase": phase_name, "error": "no_suppliers"}),
            )

        self.logger.info("Onboarding %d supplier(s)", len(suppliers_raw))

        processed_suppliers: List[Dict[str, Any]] = []
        validation_warnings: List[str] = []

        for idx, supplier in enumerate(suppliers_raw):
            supplier_name = supplier.get("supplier_name", f"Supplier_{idx}")
            country_code = supplier.get("country_code", "XX")
            supplier_id = supplier.get("supplier_id") or f"SUP-{uuid.uuid4().hex[:12]}"

            # Validate EORI if provided
            eori = supplier.get("eori_number")
            eori_valid = True
            if eori:
                eori_valid = self._validate_eori(eori)
                if not eori_valid:
                    validation_warnings.append(
                        f"Supplier '{supplier_name}': EORI '{eori}' format invalid. "
                        "Expected: 2-letter country code + 1-15 alphanumeric."
                    )

            # Validate commodities against EUDR scope
            commodities = supplier.get("commodities", [])
            valid_commodities = [c for c in commodities if c in [e.value for e in EUDRCommodity]]
            invalid_commodities = [c for c in commodities if c not in [e.value for e in EUDRCommodity]]
            if invalid_commodities:
                validation_warnings.append(
                    f"Supplier '{supplier_name}': invalid commodities {invalid_commodities}. "
                    f"EUDR covers: {[e.value for e in EUDRCommodity]}"
                )

            # Check for duplicate suppliers
            duplicate = await self._check_duplicate_supplier(supplier_name, country_code)
            if duplicate:
                validation_warnings.append(
                    f"Supplier '{supplier_name}' may be a duplicate of existing "
                    f"supplier ID '{duplicate}'."
                )

            # Determine initial country risk category
            country_risk_cat = self._classify_country_risk(country_code)

            processed_suppliers.append({
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "country_code": country_code,
                "eori_number": eori,
                "eori_valid": eori_valid,
                "contact_email": supplier.get("contact_email", ""),
                "contact_name": supplier.get("contact_name"),
                "commodities": valid_commodities,
                "certifications": supplier.get("certifications", []),
                "plot_count": supplier.get("plot_count", 0),
                "country_risk_category": country_risk_cat,
                "onboarded_at": datetime.utcnow().isoformat(),
            })

        warnings.extend(validation_warnings)

        context.state["suppliers"] = processed_suppliers
        context.state["suppliers_processed"] = len(processed_suppliers)

        outputs["suppliers_processed"] = len(processed_suppliers)
        outputs["eori_validation_issues"] = sum(
            1 for s in processed_suppliers if not s.get("eori_valid", True)
        )
        outputs["country_risk_breakdown"] = self._count_by_key(
            processed_suppliers, "country_risk_category"
        )
        outputs["commodity_breakdown"] = self._count_commodities(processed_suppliers)

        self.logger.info(
            "Phase 1 complete: %d suppliers onboarded, %d warnings",
            len(processed_suppliers), len(warnings),
        )

        provenance = self._hash({
            "phase": phase_name,
            "suppliers": len(processed_suppliers),
            "execution_id": context.execution_id,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Geolocation Collection (~3 business days typical)
    # -------------------------------------------------------------------------

    async def _phase_2_geolocation_collection(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Validate all coordinates/polygons to WGS84 6-decimal, enforce <4ha
        point vs >=4ha polygon rule, calculate areas, detect overlaps, and
        determine countries.

        Uses:
            - EUDR-002 (Geolocation Verification)
            - EUDR-006 (Plot Boundary)
            - EUDR-007 (GPS Validator)

        EUDR Rules:
            - Article 9(1)(d): Geolocation must be provided for all plots
            - Plots < 4 hectares: single point sufficient
            - Plots >= 4 hectares: full polygon boundary required
            - WGS84 datum, 6 decimal places minimum
        """
        phase_name = "geolocation_collection"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        input_data = context.state.get("input_data", {})
        geolocations_raw = input_data.get("geolocations", [])
        suppliers = context.state.get("suppliers", [])

        if not geolocations_raw:
            warnings.append(
                "No geolocation data provided. Geolocation is required per "
                "EUDR Article 9(1)(d) for all plots of production."
            )
            context.state["validated_plots"] = []
            context.state["plots_validated"] = 0

            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs={"plots_validated": 0, "geolocation_provided": False},
                warnings=warnings,
                provenance_hash=self._hash({"phase": phase_name, "plots": 0}),
            )

        self.logger.info("Validating %d geolocation record(s)", len(geolocations_raw))

        # Build supplier ID set for linkage validation
        supplier_ids = {s["supplier_id"] for s in suppliers}

        validated_plots: List[Dict[str, Any]] = []
        invalid_plots: List[Dict[str, Any]] = []

        for plot in geolocations_raw:
            plot_id = plot.get("plot_id", f"PLOT-{uuid.uuid4().hex[:8]}")
            plot_supplier = plot.get("supplier_id", "")
            area_ha = plot.get("area_hectares", 0.0)
            centroid = plot.get("centroid")
            polygon_points = plot.get("polygon_points", [])
            issues: List[str] = []

            # Validate supplier linkage
            if plot_supplier and plot_supplier not in supplier_ids:
                issues.append(
                    f"Plot supplier '{plot_supplier}' not found in onboarded suppliers"
                )

            # Validate centroid coordinates
            if centroid:
                coord_issues = self._validate_coordinate(
                    centroid.get("latitude", 0),
                    centroid.get("longitude", 0),
                )
                issues.extend(coord_issues)
            else:
                issues.append("Missing centroid coordinates")

            # Enforce <4ha point vs >=4ha polygon rule
            if area_ha >= POINT_ONLY_MAX_AREA_HA:
                if not polygon_points or len(polygon_points) < 3:
                    issues.append(
                        f"Plot area is {area_ha:.2f} ha (>= {POINT_ONLY_MAX_AREA_HA} ha). "
                        "Full polygon boundary is required per EUDR Article 9(1)(d)."
                    )
                else:
                    # Validate polygon closure and point validity
                    poly_issues = self._validate_polygon(polygon_points)
                    issues.extend(poly_issues)

            # Validate area is positive
            if area_ha <= 0:
                issues.append("Plot area must be positive")

            # Validate country code
            country = plot.get("country_code", "")
            if not country or len(country) != 2 or not country.isalpha():
                issues.append(f"Invalid country code: '{country}'")

            # Check production start date against cutoff
            prod_start = plot.get("production_start_date")
            if prod_start and prod_start > EUDR_CUTOFF_DATE:
                issues.append(
                    f"Production started {prod_start}, after EUDR cutoff "
                    f"date ({EUDR_CUTOFF_DATE}). Land may have been deforested."
                )

            plot_record = {
                "plot_id": plot_id,
                "supplier_id": plot_supplier,
                "country_code": country,
                "area_hectares": area_ha,
                "has_centroid": centroid is not None,
                "has_polygon": len(polygon_points) >= 3,
                "polygon_point_count": len(polygon_points),
                "commodity": plot.get("commodity"),
                "production_start_date": prod_start,
                "issues": issues,
                "valid": len(issues) == 0,
            }

            if issues:
                invalid_plots.append(plot_record)
                for issue in issues:
                    warnings.append(f"Plot {plot_id}: {issue}")
            else:
                validated_plots.append(plot_record)

        # Detect overlapping plots
        overlaps = self._detect_plot_overlaps(validated_plots)
        if overlaps:
            for ov in overlaps:
                warnings.append(
                    f"Potential overlap between plots {ov[0]} and {ov[1]}"
                )

        context.state["validated_plots"] = validated_plots
        context.state["invalid_plots"] = invalid_plots
        context.state["plots_validated"] = len(validated_plots)

        # Area statistics
        total_area = sum(p["area_hectares"] for p in validated_plots)
        small_plots = sum(1 for p in validated_plots if p["area_hectares"] < POINT_ONLY_MAX_AREA_HA)
        large_plots = sum(1 for p in validated_plots if p["area_hectares"] >= POINT_ONLY_MAX_AREA_HA)

        outputs["plots_validated"] = len(validated_plots)
        outputs["plots_invalid"] = len(invalid_plots)
        outputs["total_area_hectares"] = round(total_area, 4)
        outputs["small_plots_point_only"] = small_plots
        outputs["large_plots_polygon_required"] = large_plots
        outputs["overlaps_detected"] = len(overlaps)
        outputs["geolocation_provided"] = True
        outputs["country_breakdown"] = self._count_by_key(validated_plots, "country_code")

        self.logger.info(
            "Phase 2 complete: %d valid, %d invalid plots, %.2f ha total",
            len(validated_plots), len(invalid_plots), total_area,
        )

        provenance = self._hash({
            "phase": phase_name,
            "validated": len(validated_plots),
            "invalid": len(invalid_plots),
            "total_area": total_area,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Document Collection (~3 business days typical)
    # -------------------------------------------------------------------------

    async def _phase_3_document_collection(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Collect and authenticate FSC/PEFC/RSPO/ISCC/RA/UTZ certificates,
        verify permits, and check document validity periods.

        Uses:
            - EUDR-012 (Document Authentication)
            - DATA-001 (PDF Extractor)

        Steps:
            1. Collect all certification documents from input
            2. Verify each certificate against issuing body records
            3. Check validity periods (not expired)
            4. Match certifications to suppliers
            5. Score document completeness per supplier
        """
        phase_name = "document_collection"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        input_data = context.state.get("input_data", {})
        certifications_raw = input_data.get("certifications", [])
        suppliers = context.state.get("suppliers", [])

        if not certifications_raw:
            warnings.append(
                "No certifications provided. Voluntary certifications (FSC, PEFC, "
                "RSPO, ISCC, RA, UTZ) can reduce risk assessment scores."
            )
            context.state["verified_certifications"] = []
            context.state["certifications_verified"] = 0

            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs={"certifications_verified": 0, "documents_provided": False},
                warnings=warnings,
                provenance_hash=self._hash({"phase": phase_name, "certs": 0}),
            )

        self.logger.info("Processing %d certification(s)", len(certifications_raw))

        supplier_ids = {s["supplier_id"] for s in suppliers}
        now = datetime.utcnow().strftime("%Y-%m-%d")

        verified_certs: List[Dict[str, Any]] = []
        invalid_certs: List[Dict[str, Any]] = []

        for cert in certifications_raw:
            cert_id = cert.get("cert_id", f"CERT-{uuid.uuid4().hex[:8]}")
            cert_type = cert.get("cert_type", "")
            supplier_id = cert.get("supplier_id", "")
            issue_date = cert.get("issue_date", "")
            expiry_date = cert.get("expiry_date", "")
            issues: List[str] = []

            # Validate cert type
            valid_types = [ct.value for ct in CertificationType]
            if cert_type not in valid_types:
                issues.append(
                    f"Unknown certification type '{cert_type}'. "
                    f"Supported: {valid_types}"
                )

            # Validate supplier linkage
            if supplier_id and supplier_id not in supplier_ids:
                issues.append(
                    f"Certificate supplier '{supplier_id}' not found in onboarded suppliers"
                )

            # Check expiry
            if expiry_date and expiry_date < now:
                issues.append(
                    f"Certificate expired on {expiry_date}. "
                    "Expired certifications cannot reduce risk scores."
                )

            # Check issue date is before expiry
            if issue_date and expiry_date and issue_date > expiry_date:
                issues.append(
                    f"Issue date {issue_date} is after expiry date {expiry_date}"
                )

            # Verify certificate authenticity (async agent call)
            is_authentic = await self._verify_certificate(cert_id, cert_type)

            # Calculate document hash for provenance
            doc_hash = self._hash({
                "cert_id": cert_id,
                "cert_type": cert_type,
                "supplier_id": supplier_id,
                "issue_date": issue_date,
                "expiry_date": expiry_date,
            })

            cert_record = {
                "cert_id": cert_id,
                "cert_type": cert_type,
                "supplier_id": supplier_id,
                "issue_date": issue_date,
                "expiry_date": expiry_date,
                "scope": cert.get("scope", []),
                "verified": is_authentic and len(issues) == 0,
                "authentic": is_authentic,
                "issues": issues,
                "document_hash": doc_hash,
            }

            if issues:
                invalid_certs.append(cert_record)
                for issue in issues:
                    warnings.append(f"Cert {cert_id}: {issue}")
            else:
                verified_certs.append(cert_record)

        context.state["verified_certifications"] = verified_certs
        context.state["invalid_certifications"] = invalid_certs
        context.state["certifications_verified"] = len(verified_certs)

        # Calculate per-supplier certification coverage
        supplier_cert_map: Dict[str, List[str]] = {}
        for cert in verified_certs:
            sid = cert["supplier_id"]
            if sid not in supplier_cert_map:
                supplier_cert_map[sid] = []
            supplier_cert_map[sid].append(cert["cert_type"])

        suppliers_with_certs = len(supplier_cert_map)
        suppliers_without_certs = len(supplier_ids) - suppliers_with_certs

        if suppliers_without_certs > 0:
            warnings.append(
                f"{suppliers_without_certs} supplier(s) have no verified certifications."
            )

        outputs["certifications_verified"] = len(verified_certs)
        outputs["certifications_invalid"] = len(invalid_certs)
        outputs["cert_type_breakdown"] = self._count_by_key(verified_certs, "cert_type")
        outputs["suppliers_with_certs"] = suppliers_with_certs
        outputs["suppliers_without_certs"] = suppliers_without_certs
        outputs["documents_provided"] = True

        self.logger.info(
            "Phase 3 complete: %d verified, %d invalid certifications",
            len(verified_certs), len(invalid_certs),
        )

        provenance = self._hash({
            "phase": phase_name,
            "verified": len(verified_certs),
            "invalid": len(invalid_certs),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Risk Assessment (~2 business days typical)
    # -------------------------------------------------------------------------

    async def _phase_4_risk_assessment(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Calculate composite risk score from country, supplier, commodity,
        and document risk dimensions. Determine standard vs simplified DD.

        Uses:
            - EUDR-016 (Country Risk Classifier)
            - EUDR-017 (Supplier Risk Scorer)
            - EUDR-018 (Commodity Risk Analyzer)
            - EUDR-028 (Risk Assessment Coordinator)

        Risk Dimensions (weighted):
            - Country risk (30%): Article 29 benchmarking
            - Supplier risk (25%): Certification + history + data quality
            - Commodity risk (25%): Deforestation correlation by commodity
            - Document risk (20%): Document completeness + validity

        DD Type Determination:
            - All suppliers in low-risk countries + score < 30: simplified DD
            - Any supplier in high-risk country or score >= 70: standard DD
            - Otherwise: standard DD (default)
        """
        phase_name = "risk_assessment"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        suppliers = context.state.get("suppliers", [])
        validated_plots = context.state.get("validated_plots", [])
        verified_certs = context.state.get("verified_certifications", [])

        if not suppliers:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs={"error": "No suppliers available for risk assessment"},
                provenance_hash=self._hash({"phase": phase_name, "error": "no_suppliers"}),
            )

        # --- Country Risk (30% weight) ---
        country_risk = self._calculate_country_risk(suppliers)

        # --- Supplier Risk (25% weight) ---
        supplier_risk = self._calculate_supplier_risk(suppliers, verified_certs)

        # --- Commodity Risk (25% weight) ---
        commodity_risk = self._calculate_commodity_risk(suppliers)

        # --- Document Risk (20% weight) ---
        document_risk = self._calculate_document_risk(
            suppliers, verified_certs, validated_plots,
        )

        # --- Composite Score ---
        weights = {
            "country_risk": 0.30,
            "supplier_risk": 0.25,
            "commodity_risk": 0.25,
            "document_risk": 0.20,
        }

        composite = (
            country_risk * weights["country_risk"]
            + supplier_risk * weights["supplier_risk"]
            + commodity_risk * weights["commodity_risk"]
            + document_risk * weights["document_risk"]
        )
        composite = round(min(100.0, max(0.0, composite)), 2)

        # Determine risk level
        if composite >= 70.0:
            risk_level = RiskLevel.HIGH
        elif composite >= 30.0:
            risk_level = RiskLevel.STANDARD
        else:
            risk_level = RiskLevel.LOW

        # Determine DD type
        all_low_risk_countries = all(
            s.get("country_risk_category") == "low" for s in suppliers
        )
        any_high_risk_country = any(
            s.get("country_risk_category") == "high" for s in suppliers
        )

        if all_low_risk_countries and composite < 30.0:
            dd_type = DDType.SIMPLIFIED
        else:
            dd_type = DDType.STANDARD

        if any_high_risk_country:
            dd_type = DDType.STANDARD
            warnings.append(
                "At least one supplier is in a high-risk country. "
                "Standard due diligence is required per Article 29."
            )

        risk_score = {
            "country_risk": round(country_risk, 2),
            "supplier_risk": round(supplier_risk, 2),
            "commodity_risk": round(commodity_risk, 2),
            "document_risk": round(document_risk, 2),
            "composite_score": composite,
            "risk_level": risk_level.value,
            "dd_type": dd_type.value,
            "scoring_weights": weights,
        }

        context.state["risk_score"] = risk_score
        context.state["dd_type"] = dd_type.value
        context.state["risk_level"] = risk_level.value

        outputs.update(risk_score)
        outputs["suppliers_assessed"] = len(suppliers)

        if composite >= 70.0:
            warnings.append(
                f"High composite risk score ({composite:.1f}/100). "
                "Enhanced due diligence and mitigation measures are required."
            )

        self.logger.info(
            "Phase 4 complete: composite=%.2f risk_level=%s dd_type=%s",
            composite, risk_level.value, dd_type.value,
        )

        provenance = self._hash({
            "phase": phase_name,
            "composite": composite,
            "risk_level": risk_level.value,
            "dd_type": dd_type.value,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: DDS Generation (~2 business days typical)
    # -------------------------------------------------------------------------

    async def _phase_5_dds_generation(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Assemble Due Diligence Statement from all upstream data per Annex II
        requirements. Generate either standard or simplified DDS.

        Uses:
            - EUDR-030 (Documentation Generator)
            - EUDR-037 (DDS Creator)
            - EUDR-038 (Reference Number Generator)

        Annex II DDS Content:
            (a) Operator/trader information
            (b) Description of relevant commodity/product
            (c) Country of production, geolocation of all plots
            (d) Quantity/volume
            (e) Risk assessment conclusions
            (f) Risk mitigation measures taken
            (g) Declaration of compliance
        """
        phase_name = "dds_generation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        dds_id = context.state.get("dds_id", f"DDS-{uuid.uuid4().hex[:12]}")
        operator_name = context.state.get("operator_name", "")
        operator_eori = context.state.get("operator_eori")
        suppliers = context.state.get("suppliers", [])
        validated_plots = context.state.get("validated_plots", [])
        verified_certs = context.state.get("verified_certifications", [])
        risk_score = context.state.get("risk_score", {})
        dd_type = DDType(context.state.get("dd_type", "standard"))
        risk_level = context.state.get("risk_level", "standard")

        # Collect all commodities across suppliers
        all_commodities: set = set()
        for s in suppliers:
            all_commodities.update(s.get("commodities", []))

        # Build supplier details for DDS
        supplier_details = []
        for s in suppliers:
            supplier_details.append({
                "supplier_id": s["supplier_id"],
                "supplier_name": s["supplier_name"],
                "country_code": s["country_code"],
                "commodities": s.get("commodities", []),
                "country_risk_category": s.get("country_risk_category", "standard"),
            })

        # Build geolocation records for DDS
        geo_records = []
        for p in validated_plots:
            geo_records.append({
                "plot_id": p["plot_id"],
                "supplier_id": p.get("supplier_id", ""),
                "country_code": p.get("country_code", ""),
                "area_hectares": p.get("area_hectares", 0),
                "has_polygon": p.get("has_polygon", False),
                "commodity": p.get("commodity"),
            })

        # Build certification records for DDS
        cert_records = []
        for c in verified_certs:
            cert_records.append({
                "cert_id": c["cert_id"],
                "cert_type": c["cert_type"],
                "supplier_id": c["supplier_id"],
                "expiry_date": c.get("expiry_date", ""),
                "scope": c.get("scope", []),
            })

        # Generate mitigation measures based on risk level
        mitigation_measures = self._generate_mitigation_measures(
            risk_level, risk_score, suppliers,
        )

        # Generate compliance declaration
        declaration_text = self._generate_declaration(
            operator_name, dd_type, list(all_commodities),
        )

        # Assemble DDS content
        dds_content = {
            "dds_id": dds_id,
            "reference_number": None,  # Assigned on submission
            "operator_name": operator_name,
            "operator_eori": operator_eori,
            "commodities": sorted(all_commodities),
            "suppliers": supplier_details,
            "geolocations": geo_records,
            "risk_assessment": risk_score,
            "mitigation_measures": mitigation_measures,
            "certifications": cert_records,
            "dd_type": dd_type.value,
            "declaration_text": declaration_text,
            "generated_at": datetime.utcnow().isoformat(),
            "provenance_hash": "",
        }

        # Calculate DDS provenance hash
        dds_provenance = self._hash({
            "dds_id": dds_id,
            "operator_name": operator_name,
            "suppliers": [s["supplier_id"] for s in supplier_details],
            "plots": [p["plot_id"] for p in geo_records],
            "certs": [c["cert_id"] for c in cert_records],
            "risk": risk_score,
        })
        dds_content["provenance_hash"] = dds_provenance

        context.state["dds_content"] = dds_content

        # Validate DDS completeness
        completeness_issues = self._validate_dds_completeness(dds_content, dd_type)
        if completeness_issues:
            for issue in completeness_issues:
                warnings.append(f"DDS completeness: {issue}")

        outputs["dds_id"] = dds_id
        outputs["dd_type"] = dd_type.value
        outputs["commodities_covered"] = len(all_commodities)
        outputs["suppliers_included"] = len(supplier_details)
        outputs["plots_included"] = len(geo_records)
        outputs["certifications_included"] = len(cert_records)
        outputs["mitigation_measures_count"] = len(mitigation_measures)
        outputs["completeness_issues"] = len(completeness_issues)
        outputs["dds_provenance_hash"] = dds_provenance

        self.logger.info(
            "Phase 5 complete: dds_id=%s dd_type=%s %d suppliers, "
            "%d plots, %d certs",
            dds_id, dd_type.value, len(supplier_details),
            len(geo_records), len(cert_records),
        )

        provenance = self._hash({
            "phase": phase_name,
            "dds_id": dds_id,
            "dds_provenance": dds_provenance,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 6: Review and Submit (~2 business days typical)
    # -------------------------------------------------------------------------

    async def _phase_6_review_and_submit(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Human review workflow (optional auto-submit for low-risk), DDS
        validation, EU Information System submission, and archival.

        Uses:
            - EUDR-036 (EU IS Interface)
            - FOUND-005 (Citations & Evidence)

        Steps:
            1. Run final DDS validation checks
            2. Determine if auto-submit is applicable (low-risk only)
            3. Submit or queue for human review
            4. On submission: receive EU IS reference number
            5. Archive DDS with full provenance chain
        """
        phase_name = "review_and_submit"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        dds_content = context.state.get("dds_content", {})
        risk_score = context.state.get("risk_score", {})
        risk_level = context.state.get("risk_level", "standard")
        auto_submit = context.state.get("auto_submit", False)
        dds_id = dds_content.get("dds_id", "")

        # Run final DDS validation
        validation_results = await self._validate_dds_for_submission(dds_content)
        validation_passed = all(
            v.get("status") == "PASS" for v in validation_results
        )

        outputs["validation_checks"] = validation_results
        outputs["validation_passed"] = validation_passed

        if not validation_passed:
            failed_checks = [
                v["check_name"] for v in validation_results
                if v.get("status") != "PASS"
            ]
            warnings.append(
                f"DDS validation failed on: {', '.join(failed_checks)}. "
                "Review and correct before submission."
            )

        # Determine submission path
        auto_submit_eligible = (
            auto_submit
            and validation_passed
            and risk_level == RiskLevel.LOW.value
            and risk_score.get("composite_score", 100) < 30.0
        )

        submitted = False
        reference_number = None
        review_decision = None

        if auto_submit_eligible:
            # Auto-submit for low-risk DDS
            self.logger.info("Auto-submitting low-risk DDS %s", dds_id)
            submission_result = await self._submit_to_eu_is(dds_content)
            submitted = submission_result.get("submitted", False)
            reference_number = submission_result.get("reference_number")

            if submitted:
                outputs["submission_method"] = "auto_submit"
                outputs["reference_number"] = reference_number
            else:
                warnings.append(
                    "Auto-submission failed. DDS queued for manual review."
                )
                outputs["submission_method"] = "manual_review_required"
        else:
            # Queue for human review
            review_decision = await self._queue_for_review(dds_id, dds_content)
            outputs["submission_method"] = "human_review"
            outputs["review_decision"] = review_decision

            if review_decision == ReviewDecision.APPROVED.value:
                submission_result = await self._submit_to_eu_is(dds_content)
                submitted = submission_result.get("submitted", False)
                reference_number = submission_result.get("reference_number")
                outputs["reference_number"] = reference_number

        # Archive DDS
        archive_record = await self._archive_dds(
            dds_id, dds_content, submitted, reference_number,
        )

        context.state["submitted"] = submitted
        context.state["reference_number"] = reference_number

        outputs["submitted"] = submitted
        outputs["archived"] = archive_record.get("archived", False)
        outputs["archive_id"] = archive_record.get("archive_id", "")

        if submitted and reference_number:
            self.logger.info(
                "DDS %s submitted successfully, reference_number=%s",
                dds_id, reference_number,
            )
        elif not submitted:
            warnings.append(
                "DDS has not been submitted to the EU Information System. "
                "Complete review and submit before placing goods on the market."
            )

        self.logger.info(
            "Phase 6 complete: submitted=%s reference=%s",
            submitted, reference_number,
        )

        provenance = self._hash({
            "phase": phase_name,
            "dds_id": dds_id,
            "submitted": submitted,
            "reference_number": reference_number,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # =========================================================================
    # RISK CALCULATION METHODS (Zero-Hallucination, Deterministic)
    # =========================================================================

    def _calculate_country_risk(self, suppliers: List[Dict[str, Any]]) -> float:
        """
        Calculate country risk score based on Article 29 benchmarking.

        Scoring:
            High-risk country: 80 points
            Standard-risk country: 50 points
            Low-risk country: 15 points
            Average across all suppliers.
        """
        if not suppliers:
            return 50.0

        scores: List[float] = []
        for s in suppliers:
            country = s.get("country_code", "")
            if country in HIGH_RISK_COUNTRIES:
                scores.append(80.0)
            elif country in LOW_RISK_COUNTRIES:
                scores.append(15.0)
            else:
                scores.append(50.0)

        return sum(scores) / len(scores)

    def _calculate_supplier_risk(
        self,
        suppliers: List[Dict[str, Any]],
        certs: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate supplier risk based on certifications and data quality.

        Scoring per supplier (0-100, lower is better):
            - No certifications: 70 base
            - Has certifications: 30 base
            - Each valid cert: -5 (min 10)
        """
        if not suppliers:
            return 50.0

        cert_map: Dict[str, int] = {}
        for c in certs:
            sid = c.get("supplier_id", "")
            cert_map[sid] = cert_map.get(sid, 0) + 1

        scores: List[float] = []
        for s in suppliers:
            sid = s["supplier_id"]
            cert_count = cert_map.get(sid, 0)

            if cert_count == 0:
                score = 70.0
            else:
                score = max(10.0, 30.0 - (cert_count * 5.0))

            scores.append(score)

        return sum(scores) / len(scores)

    def _calculate_commodity_risk(
        self, suppliers: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate commodity risk based on deforestation correlation.

        Risk levels by commodity (empirical deforestation correlation):
            oil_palm: 85 (highest deforestation driver)
            soya: 75
            cattle: 70
            cocoa: 65
            coffee: 55
            rubber: 60
            wood: 50
        """
        commodity_risk_map: Dict[str, float] = {
            "oil_palm": 85.0,
            "soya": 75.0,
            "cattle": 70.0,
            "cocoa": 65.0,
            "rubber": 60.0,
            "coffee": 55.0,
            "wood": 50.0,
        }

        all_commodities: List[str] = []
        for s in suppliers:
            all_commodities.extend(s.get("commodities", []))

        if not all_commodities:
            return 50.0

        scores = [commodity_risk_map.get(c, 50.0) for c in all_commodities]
        return sum(scores) / len(scores)

    def _calculate_document_risk(
        self,
        suppliers: List[Dict[str, Any]],
        certs: List[Dict[str, Any]],
        plots: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate document risk based on completeness and validity.

        Scoring (0-100, lower is better):
            - Start at 80 (high risk for no documents)
            - Each verified cert: -10
            - Each validated plot: -5
            - Minimum: 5
        """
        base_score = 80.0

        cert_reduction = len(certs) * 10.0
        plot_reduction = len(plots) * 5.0

        total_reduction = cert_reduction + plot_reduction
        score = max(5.0, base_score - total_reduction)

        return score

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    def _validate_coordinate(
        self, latitude: float, longitude: float
    ) -> List[str]:
        """Validate a single coordinate pair for WGS84 compliance."""
        issues: List[str] = []

        if latitude < -90.0 or latitude > 90.0:
            issues.append(f"Latitude {latitude} out of range [-90, 90]")

        if longitude < -180.0 or longitude > 180.0:
            issues.append(f"Longitude {longitude} out of range [-180, 180]")

        # Check decimal precision (need at least 6 decimal places)
        lat_str = f"{latitude:.10f}"
        lon_str = f"{longitude:.10f}"
        lat_decimals = len(lat_str.split(".")[1].rstrip("0")) if "." in lat_str else 0
        lon_decimals = len(lon_str.split(".")[1].rstrip("0")) if "." in lon_str else 0

        if lat_decimals < MIN_COORDINATE_DECIMALS and latitude != 0.0:
            issues.append(
                f"Latitude precision insufficient: {lat_decimals} decimals "
                f"(need >= {MIN_COORDINATE_DECIMALS})"
            )
        if lon_decimals < MIN_COORDINATE_DECIMALS and longitude != 0.0:
            issues.append(
                f"Longitude precision insufficient: {lon_decimals} decimals "
                f"(need >= {MIN_COORDINATE_DECIMALS})"
            )

        return issues

    def _validate_polygon(
        self, points: List[Dict[str, Any]]
    ) -> List[str]:
        """Validate polygon boundary points."""
        issues: List[str] = []

        if len(points) < 3:
            issues.append(f"Polygon requires >= 3 points, got {len(points)}")
            return issues

        # Validate each point
        for idx, pt in enumerate(points):
            lat = pt.get("latitude", 0)
            lon = pt.get("longitude", 0)
            coord_issues = self._validate_coordinate(lat, lon)
            for ci in coord_issues:
                issues.append(f"Polygon point {idx}: {ci}")

        # Check polygon closure (first and last point should match or be close)
        if len(points) >= 3:
            first = points[0]
            last = points[-1]
            lat_diff = abs(first.get("latitude", 0) - last.get("latitude", 0))
            lon_diff = abs(first.get("longitude", 0) - last.get("longitude", 0))
            if lat_diff > 0.0001 or lon_diff > 0.0001:
                issues.append(
                    "Polygon is not closed. First and last points should match."
                )

        return issues

    def _validate_dds_completeness(
        self, dds_content: Dict[str, Any], dd_type: DDType
    ) -> List[str]:
        """Validate DDS completeness per Annex II requirements."""
        issues: List[str] = []

        # Required fields per Annex II
        if not dds_content.get("operator_name"):
            issues.append("Operator name is required (Annex II(a))")

        if not dds_content.get("commodities"):
            issues.append("At least one commodity must be specified (Annex II(b))")

        if not dds_content.get("suppliers"):
            issues.append("At least one supplier must be included (Annex II(b))")

        if not dds_content.get("geolocations"):
            issues.append("Geolocation data is required (Annex II(c))")

        if not dds_content.get("risk_assessment"):
            issues.append("Risk assessment results are required (Annex II(e))")

        if not dds_content.get("declaration_text"):
            issues.append("Compliance declaration is required (Annex II(g))")

        # Standard DD requires mitigation measures if risk > standard
        if dd_type == DDType.STANDARD:
            risk = dds_content.get("risk_assessment", {})
            if risk.get("risk_level") in ("standard", "high"):
                if not dds_content.get("mitigation_measures"):
                    issues.append(
                        "Risk mitigation measures are required for "
                        "standard/high risk (Annex II(f))"
                    )

        return issues

    async def _validate_dds_for_submission(
        self, dds_content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run final validation checks before EU IS submission."""
        await asyncio.sleep(0)
        checks: List[Dict[str, Any]] = []

        # Check 1: DDS ID present
        checks.append({
            "check_name": "dds_id_present",
            "status": "PASS" if dds_content.get("dds_id") else "FAIL",
            "message": "DDS identifier present" if dds_content.get("dds_id") else "Missing DDS ID",
        })

        # Check 2: Operator name
        checks.append({
            "check_name": "operator_name_present",
            "status": "PASS" if dds_content.get("operator_name") else "FAIL",
            "message": "Operator name present" if dds_content.get("operator_name") else "Missing operator",
        })

        # Check 3: At least one commodity
        checks.append({
            "check_name": "commodities_present",
            "status": "PASS" if dds_content.get("commodities") else "FAIL",
            "message": "Commodities specified" if dds_content.get("commodities") else "No commodities",
        })

        # Check 4: At least one supplier
        checks.append({
            "check_name": "suppliers_present",
            "status": "PASS" if dds_content.get("suppliers") else "FAIL",
            "message": "Suppliers included" if dds_content.get("suppliers") else "No suppliers",
        })

        # Check 5: Geolocation data
        checks.append({
            "check_name": "geolocations_present",
            "status": "PASS" if dds_content.get("geolocations") else "FAIL",
            "message": "Geolocations provided" if dds_content.get("geolocations") else "No geolocations",
        })

        # Check 6: Risk assessment
        checks.append({
            "check_name": "risk_assessment_present",
            "status": "PASS" if dds_content.get("risk_assessment") else "FAIL",
            "message": "Risk assessment included" if dds_content.get("risk_assessment") else "No risk assessment",
        })

        # Check 7: Declaration text
        checks.append({
            "check_name": "declaration_present",
            "status": "PASS" if dds_content.get("declaration_text") else "FAIL",
            "message": "Declaration included" if dds_content.get("declaration_text") else "No declaration",
        })

        # Check 8: Provenance hash
        checks.append({
            "check_name": "provenance_hash_present",
            "status": "PASS" if dds_content.get("provenance_hash") else "FAIL",
            "message": "Provenance hash set" if dds_content.get("provenance_hash") else "No provenance hash",
        })

        return checks

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _classify_country_risk(self, country_code: str) -> str:
        """Classify country risk level per Article 29 benchmarking."""
        if country_code in HIGH_RISK_COUNTRIES:
            return "high"
        elif country_code in LOW_RISK_COUNTRIES:
            return "low"
        return "standard"

    def _validate_eori(self, eori: str) -> bool:
        """Validate EORI number format."""
        import re
        pattern = re.compile(r"^[A-Z]{2}[A-Za-z0-9]{1,15}$")
        return bool(pattern.match(eori))

    def _detect_plot_overlaps(
        self, plots: List[Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """Detect potential overlapping plots (simplified bounding check)."""
        overlaps: List[Tuple[str, str]] = []
        # Simplified: compare plots in the same country with similar areas
        by_country: Dict[str, List[Dict[str, Any]]] = {}
        for p in plots:
            cc = p.get("country_code", "")
            if cc not in by_country:
                by_country[cc] = []
            by_country[cc].append(p)

        for cc, country_plots in by_country.items():
            for i in range(len(country_plots)):
                for j in range(i + 1, len(country_plots)):
                    # Overlap heuristic: same supplier, similar area
                    if (
                        country_plots[i].get("supplier_id") == country_plots[j].get("supplier_id")
                        and country_plots[i].get("area_hectares", 0) > 0
                        and abs(
                            country_plots[i]["area_hectares"]
                            - country_plots[j]["area_hectares"]
                        ) < 0.1
                    ):
                        overlaps.append(
                            (country_plots[i]["plot_id"], country_plots[j]["plot_id"])
                        )
        return overlaps

    def _generate_mitigation_measures(
        self,
        risk_level: str,
        risk_score: Dict[str, Any],
        suppliers: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate risk mitigation measures based on assessment results."""
        measures: List[str] = []

        if risk_level in ("standard", "high"):
            measures.append(
                "Obtain and verify sustainability certifications (FSC, PEFC, "
                "RSPO, ISCC, RA) from all suppliers."
            )
            measures.append(
                "Request satellite imagery analysis for all production plots "
                "to verify no deforestation after December 31, 2020."
            )

        if risk_score.get("country_risk", 0) >= 60:
            measures.append(
                "Engage independent third-party auditor for on-site "
                "verification of supplier operations in high-risk countries."
            )

        if risk_score.get("supplier_risk", 0) >= 50:
            measures.append(
                "Implement enhanced supplier monitoring with quarterly "
                "data updates and annual on-site audits."
            )

        if risk_score.get("commodity_risk", 0) >= 60:
            measures.append(
                "Cross-reference commodity sourcing data with Global "
                "Forest Watch deforestation alerts."
            )

        if risk_score.get("document_risk", 0) >= 60:
            measures.append(
                "Request additional documentation from suppliers: land "
                "titles, environmental permits, compliance declarations."
            )

        if risk_level == "high":
            measures.append(
                "Establish a corrective action plan with suppliers and "
                "set timeline for risk reduction milestones."
            )
            measures.append(
                "Consider alternative suppliers in lower-risk sourcing "
                "regions as part of supply chain diversification."
            )

        if not measures:
            measures.append(
                "Continue regular monitoring and maintain current "
                "compliance documentation."
            )

        return measures

    def _generate_declaration(
        self,
        operator_name: str,
        dd_type: DDType,
        commodities: List[str],
    ) -> str:
        """Generate EUDR compliance declaration text."""
        commodity_str = ", ".join(sorted(commodities)) if commodities else "relevant commodities"
        dd_label = "simplified" if dd_type == DDType.SIMPLIFIED else "standard"

        return (
            f"I, on behalf of {operator_name or 'the operator'}, hereby declare "
            f"that {dd_label} due diligence has been exercised in accordance with "
            f"EU Regulation 2023/1115 (EUDR) for the following commodities: "
            f"{commodity_str}. Based on the risk assessment conducted, the "
            f"relevant commodities and products are deforestation-free, have been "
            f"produced in accordance with the relevant legislation of the country "
            f"of production, and are covered by a due diligence statement."
        )

    @staticmethod
    def _count_by_key(
        items: List[Dict[str, Any]], key: str
    ) -> Dict[str, int]:
        """Count items by a dictionary key value."""
        counts: Dict[str, int] = {}
        for item in items:
            val = str(item.get(key, "unknown"))
            counts[val] = counts.get(val, 0) + 1
        return counts

    @staticmethod
    def _count_commodities(
        suppliers: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count commodity occurrences across suppliers."""
        counts: Dict[str, int] = {}
        for s in suppliers:
            for c in s.get("commodities", []):
                counts[c] = counts.get(c, 0) + 1
        return counts

    # =========================================================================
    # CHECKPOINT METHODS
    # =========================================================================

    def _save_checkpoint(
        self,
        context: WorkflowContext,
        phase_name: str,
        result: PhaseResult,
    ) -> None:
        """Save checkpoint for resume capability."""
        self._checkpoint_store[phase_name] = {
            "result": result.model_dump(),
            "state_snapshot": dict(context.state),
            "saved_at": datetime.utcnow().isoformat(),
        }
        context.last_checkpoint_at = datetime.utcnow()
        self.logger.debug("Checkpoint saved for phase: %s", phase_name)

    def _load_checkpoint(
        self, context: WorkflowContext, phase_name: str
    ) -> Optional[PhaseResult]:
        """Load checkpoint for a completed phase."""
        checkpoint = self._checkpoint_store.get(phase_name)
        if checkpoint and checkpoint.get("result"):
            result_data = checkpoint["result"]
            # Restore state
            if checkpoint.get("state_snapshot"):
                context.state.update(checkpoint["state_snapshot"])
            return PhaseResult(**result_data)
        return None

    def _get_resume_phase(self, context: WorkflowContext) -> Optional[str]:
        """Determine which phase to resume from based on checkpoints."""
        if not self._checkpoint_store:
            return None

        phase_order = [
            "supplier_onboarding",
            "geolocation_collection",
            "document_collection",
            "risk_assessment",
            "dds_generation",
            "review_and_submit",
        ]

        last_completed = None
        for phase in phase_order:
            cp = self._checkpoint_store.get(phase)
            if cp and cp.get("result", {}).get("status") == PhaseStatus.COMPLETED.value:
                last_completed = phase
            else:
                break

        if last_completed:
            idx = phase_order.index(last_completed)
            if idx + 1 < len(phase_order):
                return phase_order[idx + 1]

        return None

    # =========================================================================
    # ASYNC STUBS (Agent Integration Points)
    # =========================================================================

    async def _check_duplicate_supplier(
        self, name: str, country: str
    ) -> Optional[str]:
        """Check if supplier already exists. Returns existing ID or None."""
        await asyncio.sleep(0)
        return None

    async def _verify_certificate(
        self, cert_id: str, cert_type: str
    ) -> bool:
        """Verify certificate authenticity against issuing body."""
        await asyncio.sleep(0)
        return True

    async def _submit_to_eu_is(
        self, dds_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit DDS to EU Information System."""
        await asyncio.sleep(0)
        ref_num = f"EUIS-{uuid.uuid4().hex[:8].upper()}"
        return {
            "submitted": True,
            "reference_number": ref_num,
            "submitted_at": datetime.utcnow().isoformat(),
        }

    async def _queue_for_review(
        self, dds_id: str, dds_content: Dict[str, Any]
    ) -> str:
        """Queue DDS for human review. Returns review decision."""
        await asyncio.sleep(0)
        return ReviewDecision.APPROVED.value

    async def _archive_dds(
        self,
        dds_id: str,
        dds_content: Dict[str, Any],
        submitted: bool,
        reference_number: Optional[str],
    ) -> Dict[str, Any]:
        """Archive DDS with full provenance chain."""
        await asyncio.sleep(0)
        return {
            "archived": True,
            "archive_id": f"ARC-{dds_id}",
            "archived_at": datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
