# -*- coding: utf-8 -*-
"""
Supplier Onboarding Workflow
==============================

Four-phase supplier onboarding workflow for EUDR compliance. Handles supplier
data intake, profiling, geolocation setup, and initial risk scoring to prepare
suppliers for inclusion in Due Diligence Statements.

Regulatory Context:
    Per EU Regulation 2023/1115 (EUDR):
    - Article 4: Operators must ensure products placed on the EU market are
      deforestation-free and comply with country-of-origin legislation
    - Article 9: Due diligence requires identification of suppliers and
      geolocation of all plots of land
    - Article 10: Operators must collect information to demonstrate compliance
    - Article 12: Supply chain traceability throughout the value chain

    Effective supplier onboarding ensures complete and accurate data for
    DDS generation, reducing compliance risk and audit exposure.

Phases:
    1. Data intake - Import supplier data from CSV/Excel/manual entry
    2. Supplier profiling - Create profile, assess quality, check duplicates
    3. Geolocation setup - Collect and validate plot coordinates/polygons
    4. Initial risk scoring - Calculate composite risk, classify, prioritize

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
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
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class DataSource(str, Enum):
    """Source type for supplier data intake."""
    CSV = "csv"
    EXCEL = "excel"
    MANUAL = "manual"
    API = "api"
    ERP = "erp"


class EUDRCommodity(str, Enum):
    """EUDR-relevant commodities."""
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class RiskLevel(str, Enum):
    """Risk classification level."""
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class DDType(str, Enum):
    """Due diligence type."""
    STANDARD = "standard"
    SIMPLIFIED = "simplified"


class OnboardingStatus(str, Enum):
    """Supplier onboarding status."""
    APPROVED = "approved"
    PENDING_GEOLOCATION = "pending_geolocation"
    PENDING_DOCUMENTS = "pending_documents"
    PENDING_REVIEW = "pending_review"
    REJECTED = "rejected"


# Country risk benchmarking (Article 29)
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

# Maximum area for point-only geolocation
POINT_ONLY_MAX_AREA_HA = 4.0

# Required supplier fields
REQUIRED_FIELDS = ["supplier_name", "country_code", "commodity", "contact_email"]


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


class SupplierRecord(BaseModel):
    """Raw supplier data record for intake."""
    supplier_name: str = Field(..., min_length=1, description="Legal entity name")
    country_code: str = Field(..., min_length=2, max_length=2, description="ISO 3166 alpha-2")
    commodity: str = Field(..., description="Primary EUDR commodity")
    contact_email: str = Field(..., description="Primary contact email")
    contact_name: Optional[str] = Field(None)
    eori_number: Optional[str] = Field(None, description="EORI if EU-based")
    address: Optional[str] = Field(None)
    phone: Optional[str] = Field(None)
    certifications: List[str] = Field(default_factory=list)
    plot_count: int = Field(default=0, ge=0)
    additional_commodities: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(None)

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code format."""
        if not v.isalpha() or not v.isupper() or len(v) != 2:
            raise ValueError(f"Country code must be uppercase alpha-2, got: {v}")
        return v


class PlotData(BaseModel):
    """Plot geolocation data for a supplier."""
    plot_id: Optional[str] = Field(None)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    area_hectares: float = Field(default=0.0, ge=0.0)
    polygon_points: List[Dict[str, float]] = Field(default_factory=list)
    commodity: Optional[str] = Field(None)
    country_code: Optional[str] = Field(None)


class SupplierOnboardingInput(BaseModel):
    """Input data for the supplier onboarding workflow."""
    suppliers: List[SupplierRecord] = Field(
        ..., min_length=1, description="Supplier records to onboard"
    )
    plots: List[PlotData] = Field(default_factory=list, description="Plot geolocation data")
    data_source: DataSource = Field(default=DataSource.MANUAL, description="Data source type")
    config: Dict[str, Any] = Field(default_factory=dict)


class SupplierOnboardingResult(BaseModel):
    """Complete result from the supplier onboarding workflow."""
    workflow_name: str = Field(default="supplier_onboarding")
    status: PhaseStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    suppliers_processed: int = Field(default=0, ge=0)
    suppliers_approved: int = Field(default=0, ge=0)
    suppliers_pending: int = Field(default=0, ge=0)
    suppliers_rejected: int = Field(default=0, ge=0)
    plots_validated: int = Field(default=0, ge=0)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    high_risk_count: int = Field(default=0, ge=0)
    low_risk_count: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")
    execution_id: str = Field(default="")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)


# =============================================================================
# SUPPLIER ONBOARDING WORKFLOW
# =============================================================================


class SupplierOnboardingWorkflow:
    """
    Four-phase supplier onboarding workflow for EUDR compliance.

    Manages the end-to-end process of importing, profiling, validating, and
    risk-scoring suppliers before they can be included in Due Diligence
    Statements.

    Agent Dependencies:
        - DATA-001 (PDF Extractor)
        - DATA-002 (Excel/CSV Normalizer)
        - DATA-005 (EUDR Traceability Connector)
        - DATA-010 (Quality Profiler)
        - EUDR-002 (Geolocation Verification)
        - EUDR-006 (Plot Boundary)
        - EUDR-007 (GPS Validator)
        - EUDR-008 (Multi-Tier Supplier)
        - EUDR-016 (Country Risk Classifier)
        - EUDR-017 (Supplier Risk Scorer)
        - EUDR-018 (Commodity Risk Analyzer)

    Attributes:
        config: Workflow configuration.
        logger: Logger instance.
        _execution_id: Unique execution identifier.
        _phase_results: Accumulated phase results.
        _checkpoint_store: Checkpoint data for resume.

    Example:
        >>> wf = SupplierOnboardingWorkflow()
        >>> result = await wf.run(SupplierOnboardingInput(
        ...     suppliers=[SupplierRecord(
        ...         supplier_name="Amazon Soy Ltd",
        ...         country_code="BR",
        ...         commodity="soya",
        ...         contact_email="info@amazonsoy.com",
        ...     )],
        ...     data_source=DataSource.MANUAL,
        ... ))
        >>> assert result.suppliers_processed > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the SupplierOnboardingWorkflow.

        Args:
            config: Optional configuration dict.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.SupplierOnboardingWorkflow")
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []
        self._checkpoint_store: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(
        self, input_data: SupplierOnboardingInput
    ) -> SupplierOnboardingResult:
        """
        Execute the full 4-phase supplier onboarding workflow.

        Args:
            input_data: Validated input with supplier records and plot data.

        Returns:
            SupplierOnboardingResult with processing stats and risk breakdown.
        """
        started_at = datetime.utcnow()

        self.logger.info(
            "Starting supplier onboarding workflow execution_id=%s "
            "suppliers=%d plots=%d source=%s",
            self._execution_id, len(input_data.suppliers),
            len(input_data.plots), input_data.data_source.value,
        )

        context = WorkflowContext(
            execution_id=self._execution_id,
            config={**self.config, **input_data.config},
            started_at=started_at,
            state={
                "suppliers_raw": [s.model_dump() for s in input_data.suppliers],
                "plots_raw": [p.model_dump() for p in input_data.plots],
                "data_source": input_data.data_source.value,
            },
        )

        phase_handlers = [
            ("data_intake", self._phase_1_data_intake),
            ("supplier_profiling", self._phase_2_supplier_profiling),
            ("geolocation_setup", self._phase_3_geolocation_setup),
            ("initial_risk_scoring", self._phase_4_initial_risk_scoring),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase_name, handler in phase_handlers:
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

            # Save checkpoint
            self._checkpoint_store[phase_name] = {
                "result": phase_result.model_dump(),
                "state_snapshot": dict(context.state),
                "saved_at": datetime.utcnow().isoformat(),
            }
            context.last_checkpoint_at = datetime.utcnow()

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                if phase_name == "data_intake":
                    self.logger.error("Data intake failed; halting onboarding.")
                    break

        completed_at = datetime.utcnow()

        # Extract final statistics
        supplier_statuses = context.state.get("supplier_statuses", {})
        approved = sum(1 for s in supplier_statuses.values() if s == "approved")
        pending = sum(1 for s in supplier_statuses.values() if s.startswith("pending"))
        rejected = sum(1 for s in supplier_statuses.values() if s == "rejected")

        risk_levels = context.state.get("risk_levels", {})
        high_risk = sum(1 for r in risk_levels.values() if r == "high")
        low_risk = sum(1 for r in risk_levels.values() if r == "low")

        provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
        })

        self.logger.info(
            "Supplier onboarding finished execution_id=%s status=%s "
            "processed=%d approved=%d pending=%d rejected=%d",
            self._execution_id, overall_status.value,
            len(supplier_statuses), approved, pending, rejected,
        )

        return SupplierOnboardingResult(
            status=overall_status,
            phases=self._phase_results,
            suppliers_processed=context.state.get("suppliers_processed", 0),
            suppliers_approved=approved,
            suppliers_pending=pending,
            suppliers_rejected=rejected,
            plots_validated=context.state.get("plots_validated", 0),
            data_quality_score=context.state.get("data_quality_score", 0.0),
            high_risk_count=high_risk,
            low_risk_count=low_risk,
            provenance_hash=provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Data Intake
    # -------------------------------------------------------------------------

    async def _phase_1_data_intake(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Import supplier data from CSV/Excel/manual entry, parse and normalize,
        validate required fields (name, country, commodity, contact).

        Uses:
            - DATA-001 (PDF Extractor)
            - DATA-002 (Excel/CSV Normalizer)
            - DATA-005 (EUDR Traceability Connector)
        """
        phase_name = "data_intake"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        suppliers_raw = context.state.get("suppliers_raw", [])
        data_source = context.state.get("data_source", "manual")

        if not suppliers_raw:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs={"error": "No supplier records provided"},
                provenance_hash=self._hash({"phase": phase_name, "error": "no_data"}),
            )

        self.logger.info(
            "Ingesting %d supplier record(s) from %s",
            len(suppliers_raw), data_source,
        )

        parsed_suppliers: List[Dict[str, Any]] = []
        validation_errors: List[Dict[str, Any]] = []

        for idx, supplier in enumerate(suppliers_raw):
            supplier_name = supplier.get("supplier_name", "").strip()
            country_code = supplier.get("country_code", "").strip().upper()
            commodity = supplier.get("commodity", "").strip().lower()
            contact_email = supplier.get("contact_email", "").strip()

            errors: List[str] = []

            # Validate required fields
            if not supplier_name:
                errors.append("Missing required field: supplier_name")
            if not country_code or len(country_code) != 2 or not country_code.isalpha():
                errors.append(f"Invalid country_code: '{country_code}'")
            if not commodity:
                errors.append("Missing required field: commodity")
            elif commodity not in [e.value for e in EUDRCommodity]:
                errors.append(
                    f"Invalid commodity: '{commodity}'. "
                    f"Must be one of: {[e.value for e in EUDRCommodity]}"
                )
            if not contact_email or "@" not in contact_email:
                errors.append(f"Invalid contact_email: '{contact_email}'")

            # Validate EORI if provided
            eori = supplier.get("eori_number")
            if eori:
                import re
                if not re.match(r"^[A-Z]{2}[A-Za-z0-9]{1,15}$", eori):
                    warnings.append(
                        f"Supplier '{supplier_name}': EORI '{eori}' format invalid"
                    )

            if errors:
                validation_errors.append({
                    "index": idx,
                    "supplier_name": supplier_name,
                    "errors": errors,
                })
                for err in errors:
                    warnings.append(f"Row {idx}: {err}")
                continue

            # Normalize and assign ID
            supplier_id = f"SUP-{uuid.uuid4().hex[:12]}"
            all_commodities = [commodity]
            additional = supplier.get("additional_commodities", [])
            if additional:
                all_commodities.extend(
                    c.lower().strip() for c in additional
                    if c.lower().strip() in [e.value for e in EUDRCommodity]
                )

            parsed_suppliers.append({
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "country_code": country_code,
                "commodity": commodity,
                "all_commodities": list(set(all_commodities)),
                "contact_email": contact_email,
                "contact_name": supplier.get("contact_name"),
                "eori_number": eori,
                "address": supplier.get("address"),
                "phone": supplier.get("phone"),
                "certifications": supplier.get("certifications", []),
                "plot_count": supplier.get("plot_count", 0),
                "notes": supplier.get("notes"),
                "data_source": data_source,
                "ingested_at": datetime.utcnow().isoformat(),
            })

        context.state["parsed_suppliers"] = parsed_suppliers
        context.state["validation_errors"] = validation_errors

        outputs["records_received"] = len(suppliers_raw)
        outputs["records_parsed"] = len(parsed_suppliers)
        outputs["records_rejected"] = len(validation_errors)
        outputs["data_source"] = data_source
        outputs["parse_rate"] = (
            round(len(parsed_suppliers) / len(suppliers_raw) * 100, 2)
            if suppliers_raw else 0.0
        )

        if validation_errors:
            warnings.append(
                f"{len(validation_errors)} record(s) failed validation. "
                "Review and correct before re-importing."
            )

        self.logger.info(
            "Phase 1 complete: %d parsed, %d rejected out of %d records",
            len(parsed_suppliers), len(validation_errors), len(suppliers_raw),
        )

        provenance = self._hash({
            "phase": phase_name,
            "parsed": len(parsed_suppliers),
            "rejected": len(validation_errors),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Supplier Profiling
    # -------------------------------------------------------------------------

    async def _phase_2_supplier_profiling(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Create supplier profiles, assess initial data quality score, check
        for duplicates, validate EORI if EU-based, identify commodity
        categories.

        Uses:
            - EUDR-008 (Multi-Tier Supplier)
            - EUDR-017 (Supplier Risk Scorer)
            - DATA-010 (Quality Profiler)
        """
        phase_name = "supplier_profiling"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        parsed_suppliers = context.state.get("parsed_suppliers", [])

        if not parsed_suppliers:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs={"profiled": 0},
                warnings=["No suppliers to profile"],
                provenance_hash=self._hash({"phase": phase_name, "profiled": 0}),
            )

        profiled_suppliers: List[Dict[str, Any]] = []
        duplicate_candidates: List[Dict[str, Any]] = []
        total_quality_score = 0.0

        for supplier in parsed_suppliers:
            sid = supplier["supplier_id"]
            name = supplier["supplier_name"]
            country = supplier["country_code"]

            # Check for duplicates
            duplicate = await self._check_duplicate_supplier(name, country)
            if duplicate:
                duplicate_candidates.append({
                    "supplier_id": sid,
                    "supplier_name": name,
                    "potential_match": duplicate,
                })
                warnings.append(
                    f"Supplier '{name}' may be a duplicate of '{duplicate}'"
                )

            # Calculate data quality score
            quality_score = self._calculate_data_quality_score(supplier)
            total_quality_score += quality_score

            # Classify commodity categories
            commodity_categories = self._classify_commodity_categories(
                supplier.get("all_commodities", [])
            )

            # Determine country risk category
            country_risk = self._classify_country_risk(country)

            supplier["profile"] = {
                "quality_score": round(quality_score, 4),
                "completeness": self._calculate_completeness(supplier),
                "country_risk_category": country_risk,
                "commodity_categories": commodity_categories,
                "has_certifications": len(supplier.get("certifications", [])) > 0,
                "has_geolocation": supplier.get("plot_count", 0) > 0,
                "duplicate_flag": duplicate is not None,
                "profiled_at": datetime.utcnow().isoformat(),
            }

            profiled_suppliers.append(supplier)

        avg_quality = (
            total_quality_score / len(profiled_suppliers)
            if profiled_suppliers else 0.0
        )

        context.state["profiled_suppliers"] = profiled_suppliers
        context.state["data_quality_score"] = round(avg_quality, 4)
        context.state["suppliers_processed"] = len(profiled_suppliers)

        # Quality distribution
        high_quality = sum(
            1 for s in profiled_suppliers
            if s["profile"]["quality_score"] >= 0.8
        )
        medium_quality = sum(
            1 for s in profiled_suppliers
            if 0.5 <= s["profile"]["quality_score"] < 0.8
        )
        low_quality = sum(
            1 for s in profiled_suppliers
            if s["profile"]["quality_score"] < 0.5
        )

        outputs["profiled"] = len(profiled_suppliers)
        outputs["avg_quality_score"] = round(avg_quality, 4)
        outputs["quality_distribution"] = {
            "high": high_quality,
            "medium": medium_quality,
            "low": low_quality,
        }
        outputs["duplicates_detected"] = len(duplicate_candidates)
        outputs["country_risk_breakdown"] = self._count_by_key(
            [s["profile"] for s in profiled_suppliers], "country_risk_category"
        )

        if low_quality > 0:
            warnings.append(
                f"{low_quality} supplier(s) have low data quality scores (<0.5). "
                "Consider requesting additional information."
            )

        self.logger.info(
            "Phase 2 complete: %d profiled, avg_quality=%.4f, %d duplicates",
            len(profiled_suppliers), avg_quality, len(duplicate_candidates),
        )

        provenance = self._hash({
            "phase": phase_name,
            "profiled": len(profiled_suppliers),
            "avg_quality": avg_quality,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Geolocation Setup
    # -------------------------------------------------------------------------

    async def _phase_3_geolocation_setup(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Collect plot coordinates/polygons from suppliers, validate geolocation
        data, calculate areas, link plots to suppliers, check overlaps.

        Uses:
            - EUDR-002 (Geolocation Verification)
            - EUDR-006 (Plot Boundary)
            - EUDR-007 (GPS Validator)
        """
        phase_name = "geolocation_setup"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        plots_raw = context.state.get("plots_raw", [])
        profiled_suppliers = context.state.get("profiled_suppliers", [])
        supplier_ids = {s["supplier_id"] for s in profiled_suppliers}

        if not plots_raw:
            warnings.append(
                "No geolocation data provided. Geolocation is required per "
                "EUDR Article 9(1)(d). Suppliers will be marked as "
                "pending_geolocation."
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

        self.logger.info("Validating %d plot record(s)", len(plots_raw))

        validated_plots: List[Dict[str, Any]] = []
        invalid_plots: List[Dict[str, Any]] = []

        for plot in plots_raw:
            plot_id = plot.get("plot_id") or f"PLOT-{uuid.uuid4().hex[:8]}"
            lat = plot.get("latitude", 0.0)
            lon = plot.get("longitude", 0.0)
            area_ha = plot.get("area_hectares", 0.0)
            polygon_points = plot.get("polygon_points", [])
            country = plot.get("country_code", "")
            issues: List[str] = []

            # Validate coordinate range
            if lat < -90.0 or lat > 90.0:
                issues.append(f"Latitude {lat} out of range [-90, 90]")
            if lon < -180.0 or lon > 180.0:
                issues.append(f"Longitude {lon} out of range [-180, 180]")

            # Check coordinate precision (6 decimal places minimum)
            lat_decimals = len(str(lat).split(".")[-1]) if "." in str(lat) else 0
            lon_decimals = len(str(lon).split(".")[-1]) if "." in str(lon) else 0
            if lat_decimals < 6 and lat != 0.0:
                issues.append(
                    f"Latitude precision: {lat_decimals} decimals (need >= 6)"
                )
            if lon_decimals < 6 and lon != 0.0:
                issues.append(
                    f"Longitude precision: {lon_decimals} decimals (need >= 6)"
                )

            # Check area and polygon requirement
            if area_ha >= POINT_ONLY_MAX_AREA_HA:
                if not polygon_points or len(polygon_points) < 3:
                    issues.append(
                        f"Plot area {area_ha:.2f} ha >= {POINT_ONLY_MAX_AREA_HA} ha. "
                        "Polygon boundary required (Article 9(1)(d))."
                    )

            if area_ha <= 0:
                issues.append("Plot area must be positive")

            plot_record = {
                "plot_id": plot_id,
                "latitude": lat,
                "longitude": lon,
                "area_hectares": area_ha,
                "has_polygon": len(polygon_points) >= 3,
                "polygon_point_count": len(polygon_points),
                "commodity": plot.get("commodity"),
                "country_code": country,
                "issues": issues,
                "valid": len(issues) == 0,
            }

            if issues:
                invalid_plots.append(plot_record)
                for issue in issues:
                    warnings.append(f"Plot {plot_id}: {issue}")
            else:
                validated_plots.append(plot_record)

        # Check for overlapping plots
        overlaps = self._detect_overlaps(validated_plots)
        if overlaps:
            for pair in overlaps:
                warnings.append(
                    f"Potential overlap between plots {pair[0]} and {pair[1]}"
                )

        context.state["validated_plots"] = validated_plots
        context.state["invalid_plots"] = invalid_plots
        context.state["plots_validated"] = len(validated_plots)

        total_area = sum(p["area_hectares"] for p in validated_plots)

        outputs["plots_validated"] = len(validated_plots)
        outputs["plots_invalid"] = len(invalid_plots)
        outputs["total_area_hectares"] = round(total_area, 4)
        outputs["overlaps_detected"] = len(overlaps)
        outputs["geolocation_provided"] = True
        outputs["small_plots"] = sum(
            1 for p in validated_plots if p["area_hectares"] < POINT_ONLY_MAX_AREA_HA
        )
        outputs["large_plots"] = sum(
            1 for p in validated_plots if p["area_hectares"] >= POINT_ONLY_MAX_AREA_HA
        )

        self.logger.info(
            "Phase 3 complete: %d valid, %d invalid plots, %.2f ha total",
            len(validated_plots), len(invalid_plots), total_area,
        )

        provenance = self._hash({
            "phase": phase_name,
            "validated": len(validated_plots),
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
    # Phase 4: Initial Risk Scoring
    # -------------------------------------------------------------------------

    async def _phase_4_initial_risk_scoring(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Calculate initial composite risk score for each supplier, classify
        risk level, determine DD type (standard/simplified), and prioritize
        for engagement.

        Uses:
            - EUDR-016 (Country Risk Classifier)
            - EUDR-018 (Commodity Risk Analyzer)

        Risk Dimensions:
            - Country risk (40%): Article 29 benchmarking
            - Commodity risk (30%): Deforestation correlation
            - Data quality (20%): Profile completeness
            - Geolocation (10%): Plot data availability
        """
        phase_name = "initial_risk_scoring"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        profiled_suppliers = context.state.get("profiled_suppliers", [])
        validated_plots = context.state.get("validated_plots", [])

        if not profiled_suppliers:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs={"scored": 0},
                warnings=["No suppliers to score"],
                provenance_hash=self._hash({"phase": phase_name, "scored": 0}),
            )

        scored_suppliers: List[Dict[str, Any]] = []
        supplier_statuses: Dict[str, str] = {}
        risk_levels: Dict[str, str] = {}

        # Commodity risk scores
        commodity_risk_map: Dict[str, float] = {
            "oil_palm": 85.0,
            "soya": 75.0,
            "cattle": 70.0,
            "cocoa": 65.0,
            "rubber": 60.0,
            "coffee": 55.0,
            "wood": 50.0,
        }

        for supplier in profiled_suppliers:
            sid = supplier["supplier_id"]
            country = supplier["country_code"]
            commodity = supplier["commodity"]
            profile = supplier.get("profile", {})

            # Country risk (40%)
            if country in HIGH_RISK_COUNTRIES:
                country_score = 80.0
            elif country in LOW_RISK_COUNTRIES:
                country_score = 15.0
            else:
                country_score = 50.0

            # Commodity risk (30%)
            commodity_score = commodity_risk_map.get(commodity, 50.0)

            # Data quality risk (20%) - inverse of quality score
            quality_score = profile.get("quality_score", 0.5)
            data_quality_risk = (1.0 - quality_score) * 100.0

            # Geolocation risk (10%) - high if no plots
            supplier_plots = [
                p for p in validated_plots
                if p.get("supplier_id") == sid or True  # Simplified matching
            ]
            has_geo = len(supplier_plots) > 0 or supplier.get("plot_count", 0) > 0
            geo_risk = 20.0 if has_geo else 80.0

            # Composite score (deterministic weighted average)
            composite = (
                country_score * 0.40
                + commodity_score * 0.30
                + data_quality_risk * 0.20
                + geo_risk * 0.10
            )
            composite = round(min(100.0, max(0.0, composite)), 2)

            # Risk level classification
            if composite >= 70.0:
                risk_level = RiskLevel.HIGH
            elif composite >= 30.0:
                risk_level = RiskLevel.STANDARD
            else:
                risk_level = RiskLevel.LOW

            # DD type determination
            if country in LOW_RISK_COUNTRIES and composite < 30.0:
                dd_type = DDType.SIMPLIFIED
            else:
                dd_type = DDType.STANDARD

            # Onboarding status
            if not has_geo:
                status = OnboardingStatus.PENDING_GEOLOCATION
            elif len(supplier.get("certifications", [])) == 0:
                status = OnboardingStatus.PENDING_DOCUMENTS
            elif risk_level == RiskLevel.HIGH:
                status = OnboardingStatus.PENDING_REVIEW
            else:
                status = OnboardingStatus.APPROVED

            supplier["risk_assessment"] = {
                "country_risk": round(country_score, 2),
                "commodity_risk": round(commodity_score, 2),
                "data_quality_risk": round(data_quality_risk, 2),
                "geolocation_risk": round(geo_risk, 2),
                "composite_score": composite,
                "risk_level": risk_level.value,
                "dd_type": dd_type.value,
                "onboarding_status": status.value,
                "engagement_priority": self._calculate_priority(composite, country),
            }

            scored_suppliers.append(supplier)
            supplier_statuses[sid] = status.value
            risk_levels[sid] = risk_level.value

        context.state["scored_suppliers"] = scored_suppliers
        context.state["supplier_statuses"] = supplier_statuses
        context.state["risk_levels"] = risk_levels

        # Summary statistics
        high_risk = sum(1 for r in risk_levels.values() if r == "high")
        standard_risk = sum(1 for r in risk_levels.values() if r == "standard")
        low_risk = sum(1 for r in risk_levels.values() if r == "low")

        avg_composite = (
            sum(
                s["risk_assessment"]["composite_score"]
                for s in scored_suppliers
            ) / len(scored_suppliers)
            if scored_suppliers else 0.0
        )

        outputs["scored"] = len(scored_suppliers)
        outputs["avg_composite_score"] = round(avg_composite, 2)
        outputs["risk_distribution"] = {
            "high": high_risk,
            "standard": standard_risk,
            "low": low_risk,
        }
        outputs["dd_type_distribution"] = {
            "standard": sum(
                1 for s in scored_suppliers
                if s["risk_assessment"]["dd_type"] == "standard"
            ),
            "simplified": sum(
                1 for s in scored_suppliers
                if s["risk_assessment"]["dd_type"] == "simplified"
            ),
        }
        outputs["status_distribution"] = {}
        for st in OnboardingStatus:
            count = sum(1 for s in supplier_statuses.values() if s == st.value)
            if count > 0:
                outputs["status_distribution"][st.value] = count

        if high_risk > 0:
            warnings.append(
                f"{high_risk} supplier(s) classified as HIGH risk. "
                "Enhanced due diligence and mitigation measures required."
            )

        self.logger.info(
            "Phase 4 complete: %d scored, avg=%.2f, high=%d standard=%d low=%d",
            len(scored_suppliers), avg_composite, high_risk, standard_risk, low_risk,
        )

        provenance = self._hash({
            "phase": phase_name,
            "scored": len(scored_suppliers),
            "avg_composite": avg_composite,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_data_quality_score(
        self, supplier: Dict[str, Any]
    ) -> float:
        """Calculate data quality score (0-1) based on field completeness."""
        total_fields = 12
        populated = 0

        check_fields = [
            "supplier_name", "country_code", "commodity", "contact_email",
            "contact_name", "eori_number", "address", "phone",
            "certifications", "plot_count", "notes", "additional_commodities",
        ]

        for field in check_fields:
            value = supplier.get(field)
            if value and value != 0 and value != []:
                populated += 1

        return populated / total_fields

    def _calculate_completeness(self, supplier: Dict[str, Any]) -> float:
        """Calculate required field completeness (0-1)."""
        required = ["supplier_name", "country_code", "commodity", "contact_email"]
        populated = sum(1 for f in required if supplier.get(f))
        return populated / len(required) if required else 0.0

    def _classify_country_risk(self, country_code: str) -> str:
        """Classify country risk category."""
        if country_code in HIGH_RISK_COUNTRIES:
            return "high"
        elif country_code in LOW_RISK_COUNTRIES:
            return "low"
        return "standard"

    def _classify_commodity_categories(
        self, commodities: List[str]
    ) -> List[str]:
        """Classify commodities into EUDR categories."""
        categories: List[str] = []
        agriculture = {"cattle", "soya", "oil_palm", "cocoa", "coffee"}
        forestry = {"wood", "rubber"}

        if any(c in agriculture for c in commodities):
            categories.append("agriculture")
        if any(c in forestry for c in commodities):
            categories.append("forestry")

        return categories

    def _calculate_priority(
        self, composite_score: float, country_code: str
    ) -> int:
        """
        Calculate engagement priority (1=highest, 5=lowest).

        Higher risk and high-risk country suppliers get priority.
        """
        if composite_score >= 80.0 and country_code in HIGH_RISK_COUNTRIES:
            return 1
        elif composite_score >= 70.0:
            return 2
        elif composite_score >= 50.0:
            return 3
        elif composite_score >= 30.0:
            return 4
        return 5

    def _detect_overlaps(
        self, plots: List[Dict[str, Any]]
    ) -> List[tuple]:
        """Detect potential overlapping plots (simplified)."""
        overlaps = []
        for i in range(len(plots)):
            for j in range(i + 1, len(plots)):
                pi = plots[i]
                pj = plots[j]
                # Simple proximity check
                lat_diff = abs(pi.get("latitude", 0) - pj.get("latitude", 0))
                lon_diff = abs(pi.get("longitude", 0) - pj.get("longitude", 0))
                if lat_diff < 0.001 and lon_diff < 0.001:
                    overlaps.append((pi["plot_id"], pj["plot_id"]))
        return overlaps

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

    # =========================================================================
    # ASYNC STUBS
    # =========================================================================

    async def _check_duplicate_supplier(
        self, name: str, country: str
    ) -> Optional[str]:
        """Check if supplier already exists. Returns existing ID or None."""
        await asyncio.sleep(0)
        return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
