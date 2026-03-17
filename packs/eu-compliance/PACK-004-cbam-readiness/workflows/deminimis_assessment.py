# -*- coding: utf-8 -*-
"""
De Minimis Assessment Workflow
================================

Three-phase annual assessment workflow for CBAM de minimis threshold
evaluation. Projects import volumes, compares against regulatory thresholds,
and determines exemption eligibility.

Regulatory Context:
    Per EU CBAM Regulation 2023/956:
    - Article 2(5): De minimis thresholds exempt small importers from
      CBAM obligations if total imports per sector group fall below
      specified volume thresholds
    - Omnibus Simplification Package COM(2025) 508: Raised the de minimis
      threshold from 150 EUR to potentially higher levels and introduced
      a 50-tonne threshold per sector group per year

    De minimis thresholds (per sector group, per year):
        - Each CBAM sector (cement, iron/steel, aluminium, fertilisers,
          electricity, hydrogen) has an independent 50-tonne threshold
        - If total imports in a sector are below 50 tonnes/year, the
          importer is exempt from CBAM certificate obligations for
          that sector
        - The exemption applies per sector, not across all sectors combined

    Assessment timing:
        - Should be performed annually, ideally Q4 of the current year
          to project next year's obligations
        - Can be revisited quarterly if import patterns change

Phases:
    1. Volume projection - Project annual import volumes by sector group
    2. Threshold analysis - Compare projections against 50-tonne thresholds
    3. Exemption determination - Issue or revoke de minimis exemptions

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ExemptionStatus(str, Enum):
    """De minimis exemption status for a sector."""
    EXEMPT = "exempt"            # Below threshold, no CBAM obligation
    NOT_EXEMPT = "not_exempt"    # Above threshold, full obligation
    BORDERLINE = "borderline"    # Within 10% of threshold, monitor closely
    REVOKED = "revoked"          # Previously exempt, now exceeds threshold


class ProjectionMethod(str, Enum):
    """Volume projection methodology."""
    HISTORICAL_AVERAGE = "historical_average"
    TREND_EXTRAPOLATION = "trend_extrapolation"
    SEASONAL_ADJUSTED = "seasonal_adjusted"
    BUDGET_BASED = "budget_based"


# =============================================================================
# CONSTANTS
# =============================================================================

# De minimis threshold in tonnes per sector per year
DEMINIMIS_THRESHOLD_TONNES = 50.0

# Borderline percentage (within 10% of threshold triggers monitoring)
BORDERLINE_PCT = 10.0

# CBAM sector groups
CBAM_SECTORS = ["cement", "iron_steel", "aluminium", "fertilisers", "electricity", "hydrogen"]


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


class ImportHistoryRecord(BaseModel):
    """Historical import record for volume projection."""
    year: int = Field(..., ge=2020, description="Year of import")
    quarter: Optional[str] = Field(None, description="Quarter if available")
    sector: str = Field(..., description="CBAM sector")
    quantity_tonnes: float = Field(..., ge=0, description="Import quantity in tonnes")
    country_of_origin: Optional[str] = Field(None)
    cn_codes: List[str] = Field(default_factory=list)


class SectorProjection(BaseModel):
    """Projected annual import volume for a single sector."""
    sector: str = Field(..., description="CBAM sector")
    projected_volume_tonnes: float = Field(default=0.0, ge=0)
    historical_volumes: Dict[int, float] = Field(
        default_factory=dict, description="Historical volumes by year"
    )
    growth_rate_pct: float = Field(default=0.0, description="Projected growth rate")
    confidence_level: float = Field(default=0.0, ge=0, le=1.0)
    projection_method: ProjectionMethod = Field(default=ProjectionMethod.HISTORICAL_AVERAGE)
    threshold_tonnes: float = Field(default=DEMINIMIS_THRESHOLD_TONNES)
    below_threshold: bool = Field(default=False)
    margin_tonnes: float = Field(default=0.0, description="Distance from threshold (negative = below)")


class SectorExemption(BaseModel):
    """De minimis exemption determination for a single sector."""
    sector: str = Field(...)
    exemption_status: ExemptionStatus = Field(...)
    projected_volume_tonnes: float = Field(default=0.0, ge=0)
    threshold_tonnes: float = Field(default=DEMINIMIS_THRESHOLD_TONNES)
    margin_tonnes: float = Field(default=0.0)
    margin_pct: float = Field(default=0.0, description="Margin as % of threshold")
    recommendation: str = Field(default="")
    monitoring_required: bool = Field(default=False)
    effective_date: Optional[str] = Field(None)
    expiry_date: Optional[str] = Field(None)


class DeMinimisResult(BaseModel):
    """Complete result from the de minimis assessment workflow."""
    workflow_name: str = Field(default="deminimis_assessment")
    status: PhaseStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    assessment_id: str = Field(..., description="Assessment identifier")
    assessment_year: int = Field(..., description="Year being assessed")
    sectors_assessed: int = Field(default=0, ge=0)
    sectors_exempt: int = Field(default=0, ge=0)
    sectors_not_exempt: int = Field(default=0, ge=0)
    sectors_borderline: int = Field(default=0, ge=0)
    sector_projections: List[SectorProjection] = Field(default_factory=list)
    sector_exemptions: List[SectorExemption] = Field(default_factory=list)
    fully_exempt: bool = Field(default=False, description="Exempt across ALL sectors")
    total_projected_volume_tonnes: float = Field(default=0.0, ge=0)
    estimated_avoided_cost_eur: float = Field(default=0.0, ge=0)
    provenance_hash: str = Field(default="")
    execution_id: str = Field(default="")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)


# =============================================================================
# DE MINIMIS ASSESSMENT WORKFLOW
# =============================================================================


class DeMinimisAssessmentWorkflow:
    """
    Three-phase annual de minimis threshold assessment workflow.

    Evaluates whether an importer qualifies for de minimis exemptions
    in each CBAM sector based on projected annual import volumes.

    Importers below the 50-tonne threshold per sector are exempt from
    CBAM certificate obligations for that sector, significantly reducing
    compliance costs and administrative burden.

    Attributes:
        config: Optional configuration dict.
        logger: Logger instance.
        _execution_id: Unique execution identifier.
        _phase_results: Accumulated phase results.

    Example:
        >>> wf = DeMinimisAssessmentWorkflow()
        >>> result = await wf.execute(
        ...     config={"ets_price_eur": 80.0},
        ...     import_history=[
        ...         ImportHistoryRecord(year=2025, sector="cement", quantity_tonnes=30),
        ...     ],
        ... )
        >>> assert result.sectors_exempt > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the DeMinimisAssessmentWorkflow.

        Args:
            config: Optional configuration dict.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.DeMinimisAssessmentWorkflow")
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        config: Optional[Dict[str, Any]],
        import_history: List[ImportHistoryRecord],
    ) -> DeMinimisResult:
        """
        Execute the full 3-phase de minimis assessment workflow.

        Args:
            config: Execution-level config overrides.
            import_history: Historical import records for volume projection.

        Returns:
            DeMinimisResult with sector-by-sector exemption determinations.
        """
        started_at = datetime.utcnow()
        merged_config = {**self.config, **(config or {})}
        assessment_year = merged_config.get(
            "assessment_year", datetime.utcnow().year + 1
        )
        assessment_id = f"DM-{assessment_year}-{self._execution_id[:8]}"

        self.logger.info(
            "Starting de minimis assessment execution_id=%s year=%d",
            self._execution_id, assessment_year,
        )

        context: Dict[str, Any] = {
            "config": merged_config,
            "import_history": import_history,
            "assessment_year": assessment_year,
            "assessment_id": assessment_id,
            "execution_id": self._execution_id,
        }

        phase_handlers = [
            ("volume_projection", self._phase_1_volume_projection),
            ("threshold_analysis", self._phase_2_threshold_analysis),
            ("exemption_determination", self._phase_3_exemption_determination),
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
                self.logger.error("Phase '%s' failed: %s", phase_name, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    outputs={"error": str(exc)},
                    provenance_hash=self._hash({"error": str(exc)}),
                )

            self._phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                if phase_name == "volume_projection":
                    break

        completed_at = datetime.utcnow()

        # Extract results
        projections = context.get("sector_projections", [])
        exemptions = context.get("sector_exemptions", [])
        sectors_exempt = sum(1 for e in exemptions if e.exemption_status == ExemptionStatus.EXEMPT)
        sectors_not_exempt = sum(1 for e in exemptions if e.exemption_status == ExemptionStatus.NOT_EXEMPT)
        sectors_borderline = sum(1 for e in exemptions if e.exemption_status == ExemptionStatus.BORDERLINE)
        total_volume = sum(p.projected_volume_tonnes for p in projections)
        fully_exempt = sectors_exempt == len(exemptions) and len(exemptions) > 0

        provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
            "exempt": sectors_exempt,
        })

        self.logger.info(
            "De minimis assessment finished: %d exempt, %d not_exempt, %d borderline",
            sectors_exempt, sectors_not_exempt, sectors_borderline,
        )

        return DeMinimisResult(
            status=overall_status,
            phases=self._phase_results,
            assessment_id=assessment_id,
            assessment_year=assessment_year,
            sectors_assessed=len(projections),
            sectors_exempt=sectors_exempt,
            sectors_not_exempt=sectors_not_exempt,
            sectors_borderline=sectors_borderline,
            sector_projections=projections,
            sector_exemptions=exemptions,
            fully_exempt=fully_exempt,
            total_projected_volume_tonnes=round(total_volume, 4),
            estimated_avoided_cost_eur=context.get("estimated_avoided_cost_eur", 0.0),
            provenance_hash=provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Volume Projection
    # -------------------------------------------------------------------------

    async def _phase_1_volume_projection(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Project annual import volumes by sector group from historical data.

        Uses historical import records to project next year's volumes.
        Supports multiple projection methods:
            - Historical average (default): Mean of last 3 years
            - Trend extrapolation: Linear trend from historical data
            - Seasonal adjusted: Account for quarterly seasonality
            - Budget based: Use budget/procurement plan data
        """
        phase_name = "volume_projection"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        import_history: List[ImportHistoryRecord] = context.get("import_history", [])
        assessment_year = context["assessment_year"]
        projection_method_str = context["config"].get(
            "projection_method", ProjectionMethod.HISTORICAL_AVERAGE.value
        )
        projection_method = ProjectionMethod(projection_method_str)

        # Aggregate historical data by sector and year
        sector_year_volumes: Dict[str, Dict[int, float]] = {}
        for record in import_history:
            sector = record.sector
            if sector not in sector_year_volumes:
                sector_year_volumes[sector] = {}
            year_vol = sector_year_volumes[sector]
            year_vol[record.year] = year_vol.get(record.year, 0) + record.quantity_tonnes

        # Project volumes for each CBAM sector
        projections: List[SectorProjection] = []

        for sector in CBAM_SECTORS:
            historical = sector_year_volumes.get(sector, {})

            if not historical:
                # No history for this sector: project zero
                projections.append(SectorProjection(
                    sector=sector,
                    projected_volume_tonnes=0.0,
                    historical_volumes={},
                    growth_rate_pct=0.0,
                    confidence_level=0.0,
                    projection_method=projection_method,
                    threshold_tonnes=DEMINIMIS_THRESHOLD_TONNES,
                    below_threshold=True,
                    margin_tonnes=-DEMINIMIS_THRESHOLD_TONNES,
                ))
                continue

            # Calculate projection based on method
            projected, growth, confidence = self._project_volume(
                historical, assessment_year, projection_method,
            )

            margin = projected - DEMINIMIS_THRESHOLD_TONNES

            projections.append(SectorProjection(
                sector=sector,
                projected_volume_tonnes=round(projected, 4),
                historical_volumes=historical,
                growth_rate_pct=round(growth, 2),
                confidence_level=round(confidence, 4),
                projection_method=projection_method,
                threshold_tonnes=DEMINIMIS_THRESHOLD_TONNES,
                below_threshold=projected < DEMINIMIS_THRESHOLD_TONNES,
                margin_tonnes=round(margin, 4),
            ))

        # Warn on sectors with limited history
        for proj in projections:
            if 0 < len(proj.historical_volumes) < 2:
                warnings.append(
                    f"Sector '{proj.sector}': only {len(proj.historical_volumes)} year(s) "
                    "of history; projection confidence is low"
                )

        context["sector_projections"] = projections

        outputs["sectors_projected"] = len(projections)
        outputs["projection_method"] = projection_method.value
        outputs["projections"] = [p.model_dump() for p in projections]

        self.logger.info(
            "Phase 1 complete: %d sectors projected using %s method",
            len(projections), projection_method.value,
        )

        provenance = self._hash({
            "phase": phase_name,
            "sectors": len(projections),
            "method": projection_method.value,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Threshold Analysis
    # -------------------------------------------------------------------------

    async def _phase_2_threshold_analysis(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Compare projected volumes against 50-tonne thresholds per sector.

        For each sector:
            - Below threshold: eligible for de minimis exemption
            - Above threshold: full CBAM obligation applies
            - Borderline (within 10%): requires enhanced monitoring
        """
        phase_name = "threshold_analysis"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        projections: List[SectorProjection] = context.get("sector_projections", [])

        analysis_results: List[Dict[str, Any]] = []
        below_count = 0
        above_count = 0
        borderline_count = 0

        for proj in projections:
            volume = proj.projected_volume_tonnes
            threshold = proj.threshold_tonnes
            margin = volume - threshold
            margin_pct = (margin / threshold * 100) if threshold > 0 else 0.0

            if volume == 0:
                status = "no_imports"
                below_count += 1
            elif volume < threshold * (1 - BORDERLINE_PCT / 100):
                status = "clearly_below"
                below_count += 1
            elif volume < threshold:
                status = "borderline_below"
                borderline_count += 1
                warnings.append(
                    f"Sector '{proj.sector}': projected volume {volume:.2f}t is within "
                    f"{BORDERLINE_PCT}% of threshold ({threshold}t). Monitor closely."
                )
            elif volume < threshold * (1 + BORDERLINE_PCT / 100):
                status = "borderline_above"
                borderline_count += 1
                warnings.append(
                    f"Sector '{proj.sector}': projected volume {volume:.2f}t is just above "
                    f"threshold ({threshold}t). Consider volume reduction strategies."
                )
            else:
                status = "clearly_above"
                above_count += 1

            analysis_results.append({
                "sector": proj.sector,
                "projected_volume_tonnes": volume,
                "threshold_tonnes": threshold,
                "margin_tonnes": round(margin, 4),
                "margin_pct": round(margin_pct, 2),
                "status": status,
            })

        outputs["analysis_results"] = analysis_results
        outputs["below_threshold"] = below_count
        outputs["above_threshold"] = above_count
        outputs["borderline"] = borderline_count

        self.logger.info(
            "Phase 2 complete: %d below, %d above, %d borderline",
            below_count, above_count, borderline_count,
        )

        provenance = self._hash({
            "phase": phase_name,
            "below": below_count,
            "above": above_count,
            "borderline": borderline_count,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Exemption Determination
    # -------------------------------------------------------------------------

    async def _phase_3_exemption_determination(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Issue or revoke de minimis exemptions per sector.

        Generates exemption certificates for sectors below threshold,
        flags sectors requiring monitoring, and calculates the estimated
        cost avoidance from exemptions.
        """
        phase_name = "exemption_determination"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        projections: List[SectorProjection] = context.get("sector_projections", [])
        assessment_year = context["assessment_year"]
        ets_price = context["config"].get("ets_price_eur", 80.0)

        # Check existing exemptions (for revocation tracking)
        existing_exemptions = await self._fetch_existing_exemptions(context)

        exemptions: List[SectorExemption] = []
        total_avoided_cost = Decimal("0")

        for proj in projections:
            volume = proj.projected_volume_tonnes
            threshold = proj.threshold_tonnes
            margin = volume - threshold
            margin_pct = (margin / threshold * 100) if threshold > 0 else 0.0

            # Determine exemption status
            if volume < threshold:
                if abs(margin_pct) < BORDERLINE_PCT:
                    status = ExemptionStatus.BORDERLINE
                    recommendation = (
                        f"Monitor imports closely. Current projection ({volume:.2f}t) is within "
                        f"{BORDERLINE_PCT}% of the {threshold:.0f}t threshold."
                    )
                    monitoring_required = True
                else:
                    status = ExemptionStatus.EXEMPT
                    recommendation = (
                        f"De minimis exemption granted. Projected volume ({volume:.2f}t) "
                        f"is well below the {threshold:.0f}t threshold."
                    )
                    monitoring_required = False

                    # Calculate avoided cost (approximate)
                    # Assume 1.5 tCO2e/t average embedded emissions
                    avoided_emissions = Decimal(str(volume)) * Decimal("1.5")
                    avoided_cost = (avoided_emissions * Decimal(str(ets_price))).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    total_avoided_cost += avoided_cost
            else:
                # Check if previously exempt (revocation)
                previously_exempt = proj.sector in existing_exemptions
                if previously_exempt:
                    status = ExemptionStatus.REVOKED
                    recommendation = (
                        f"De minimis exemption REVOKED. Projected volume ({volume:.2f}t) "
                        f"now exceeds the {threshold:.0f}t threshold. "
                        "Full CBAM obligations apply."
                    )
                    warnings.append(
                        f"Sector '{proj.sector}': de minimis exemption revoked. "
                        "Register as authorized CBAM declarant for this sector."
                    )
                else:
                    status = ExemptionStatus.NOT_EXEMPT
                    recommendation = (
                        f"Not eligible for de minimis exemption. "
                        f"Projected volume ({volume:.2f}t) exceeds {threshold:.0f}t threshold."
                    )
                monitoring_required = True

            exemptions.append(SectorExemption(
                sector=proj.sector,
                exemption_status=status,
                projected_volume_tonnes=volume,
                threshold_tonnes=threshold,
                margin_tonnes=round(margin, 4),
                margin_pct=round(margin_pct, 2),
                recommendation=recommendation,
                monitoring_required=monitoring_required,
                effective_date=f"{assessment_year}-01-01",
                expiry_date=f"{assessment_year}-12-31",
            ))

        context["sector_exemptions"] = exemptions
        context["estimated_avoided_cost_eur"] = float(total_avoided_cost)

        outputs["exemptions"] = [e.model_dump() for e in exemptions]
        outputs["estimated_avoided_cost_eur"] = float(total_avoided_cost)
        outputs["fully_exempt"] = all(
            e.exemption_status in (ExemptionStatus.EXEMPT, ExemptionStatus.BORDERLINE)
            for e in exemptions
        )

        self.logger.info(
            "Phase 3 complete: %d exemptions, avoided cost=%.2f EUR",
            sum(1 for e in exemptions if e.exemption_status == ExemptionStatus.EXEMPT),
            float(total_avoided_cost),
        )

        provenance = self._hash({
            "phase": phase_name,
            "exemptions": len(exemptions),
            "avoided_cost": float(total_avoided_cost),
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

    def _project_volume(
        self,
        historical: Dict[int, float],
        target_year: int,
        method: ProjectionMethod,
    ) -> Tuple[float, float, float]:
        """
        Project volume for the target year from historical data.

        Returns:
            Tuple of (projected_volume, growth_rate_pct, confidence_level).
        """
        years = sorted(historical.keys())
        values = [historical[y] for y in years]

        if not values:
            return 0.0, 0.0, 0.0

        if method == ProjectionMethod.HISTORICAL_AVERAGE:
            # Use last 3 years average
            recent = values[-3:] if len(values) >= 3 else values
            projected = sum(recent) / len(recent)
            confidence = min(1.0, len(recent) / 3.0) * 0.7
            growth = 0.0
            if len(recent) >= 2 and recent[0] > 0:
                growth = ((recent[-1] - recent[0]) / recent[0]) * 100 / max(1, len(recent) - 1)
            return round(projected, 4), round(growth, 2), round(confidence, 4)

        elif method == ProjectionMethod.TREND_EXTRAPOLATION:
            # Simple linear trend
            if len(values) < 2:
                return values[0], 0.0, 0.3

            n = len(values)
            x_mean = sum(range(n)) / n
            y_mean = sum(values) / n
            numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                return y_mean, 0.0, 0.5

            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            # Project to target year
            years_ahead = target_year - years[-1]
            projected = intercept + slope * (n - 1 + years_ahead)
            projected = max(0.0, projected)  # Cannot be negative

            growth = (slope / y_mean * 100) if y_mean > 0 else 0.0
            confidence = min(1.0, n / 5.0) * 0.8

            return round(projected, 4), round(growth, 2), round(confidence, 4)

        else:
            # Default to average for other methods
            projected = sum(values) / len(values)
            return round(projected, 4), 0.0, 0.5

    # =========================================================================
    # ASYNC STUBS
    # =========================================================================

    async def _fetch_existing_exemptions(
        self, context: Dict[str, Any]
    ) -> set:
        """Fetch sectors with existing de minimis exemptions."""
        await asyncio.sleep(0)
        return set()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
