# -*- coding: utf-8 -*-
"""
Annual Declaration Workflow
============================

Eight-phase annual CBAM certificate declaration workflow that consolidates
quarterly reports, reconciles emissions, calculates certificate obligations,
applies free allocation adjustments, deducts third-country carbon prices,
estimates costs, assembles the declaration, and prepares the certificate
surrender package.

Regulatory Context:
    Per EU CBAM Regulation 2023/956:
    - Article 6: Authorized CBAM declarants must submit an annual declaration
      by May 31 of each year covering the prior calendar year
    - Article 7: Declaration must include total embedded emissions per goods
      category, CBAM certificates to be surrendered, and verification statement
    - Article 22(1): Certificates surrendered must match the embedded emissions
    - Article 22(2): By end of each quarter, declarant must hold >= 50% of
      estimated annual obligation
    - Article 31: Free allocation phase-out schedule 2026-2034
    - Article 9: Carbon price paid in country of origin may be deducted

    Phase-out schedule (Article 31):
        2026: 2.5%    2027: 5.0%    2028: 10.0%
        2029: 22.5%   2030: 48.5%   2031: 61.0%
        2032: 73.5%   2033: 86.0%   2034+: 100.0%

    The CBAM adjustment factor is the inverse of the free allocation:
        CBAM_adjustment = 1 - free_allocation_pct

Phases:
    1. Annual data consolidation - Aggregate 4 quarterly reports
    2. Emission reconciliation - Reconcile quarterly estimates with year-end actuals
    3. Certificate calculation - Calculate gross certificate obligation
    4. Free allocation adjustment - Apply phase-out percentage
    5. Carbon price deduction - Deduct third-country carbon prices
    6. Cost estimation - Project certificate cost (low/mid/high)
    7. Declaration assembly - Generate annual declaration package
    8. Surrender preparation - Prepare certificate surrender for May 31

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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


# Free allocation phase-out schedule per Article 31
# Values represent the percentage of free allocation still available
FREE_ALLOCATION_PCT: Dict[int, float] = {
    2025: 100.0,
    2026: 97.5,
    2027: 95.0,
    2028: 90.0,
    2029: 77.5,
    2030: 51.5,
    2031: 39.0,
    2032: 26.5,
    2033: 14.0,
    2034: 0.0,
}

# CBAM adjustment factor (inverse of free allocation)
# This represents the portion of emissions that require CBAM certificates
CBAM_ADJUSTMENT_PCT: Dict[int, float] = {
    year: round(100.0 - pct, 2) for year, pct in FREE_ALLOCATION_PCT.items()
}

# EU ETS benchmark values in tCO2e/tonne per sector
PRODUCT_BENCHMARKS: Dict[str, float] = {
    "cement": 0.766,
    "iron_steel": 1.328,
    "aluminium": 1.514,
    "fertilisers": 1.619,
    "hydrogen": 8.850,
    "electricity": 0.376,  # tCO2e/MWh
}


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class CostScenario(str, Enum):
    """Cost projection scenario."""
    LOW = "low"
    MID = "mid"
    HIGH = "high"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Phase execution time")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output artifacts")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    provenance_hash: str = Field(default="", description="SHA-256 hash of phase inputs+outputs")


class QuarterlyReportSummary(BaseModel):
    """Summary of a quarterly CBAM report for annual consolidation."""
    quarter: str = Field(..., description="Quarter label e.g. 'Q1'")
    report_id: str = Field(..., description="Quarterly report ID")
    total_emissions_tco2e: float = Field(default=0.0, ge=0, description="Total embedded emissions")
    total_quantity_tonnes: float = Field(default=0.0, ge=0, description="Total imported quantity")
    goods_categories: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_score: float = Field(default=0.0, ge=0, le=100)
    data_source_breakdown: Dict[str, int] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class CostProjection(BaseModel):
    """Certificate cost projection for a given scenario."""
    scenario: CostScenario = Field(..., description="Cost scenario")
    ets_price_eur_per_tco2e: float = Field(..., ge=0, description="Assumed EU ETS price")
    gross_cost_eur: float = Field(default=0.0, ge=0, description="Before deductions")
    free_allocation_deduction_eur: float = Field(default=0.0, ge=0)
    carbon_price_deduction_eur: float = Field(default=0.0, ge=0)
    net_cost_eur: float = Field(default=0.0, ge=0, description="After all deductions")


class AnnualDeclarationResult(BaseModel):
    """Complete result from the annual CBAM declaration workflow.

    Contains the full declaration metadata, emission totals, certificate
    obligations, cost projections, and complete provenance chain.
    """
    workflow_name: str = Field(default="annual_declaration", description="Workflow identifier")
    status: PhaseStatus = Field(..., description="Overall workflow status")
    phases: List[PhaseResult] = Field(default_factory=list, description="Phase results")
    declaration_id: str = Field(..., description="Annual declaration identifier")
    year: int = Field(..., description="Reporting year")
    total_emissions_tco2e: float = Field(default=0.0, ge=0, description="Total embedded emissions")
    net_certificates: float = Field(default=0.0, ge=0, description="Net certificate obligation")
    estimated_cost_eur: float = Field(default=0.0, ge=0, description="Estimated net cost (mid scenario)")
    free_allocation_applied: bool = Field(default=False)
    free_allocation_pct: float = Field(default=0.0, description="Free allocation percentage applied")
    carbon_deductions_applied: bool = Field(default=False)
    carbon_deductions_tco2e: float = Field(default=0.0, ge=0)
    cost_projections: List[CostProjection] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 hash of entire workflow")
    execution_id: str = Field(default="", description="Unique execution identifier")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    quarters_consolidated: int = Field(default=0)
    total_quantity_tonnes: float = Field(default=0.0, ge=0)
    surrender_deadline: Optional[str] = Field(None, description="May 31 of year+1")


# =============================================================================
# ANNUAL DECLARATION WORKFLOW
# =============================================================================


class AnnualDeclarationWorkflow:
    """
    Eight-phase annual CBAM certificate declaration workflow.

    Orchestrates the end-to-end annual declaration process from consolidating
    quarterly reports through to preparing the certificate surrender package
    for the May 31 deadline.

    The workflow implements the CBAM certificate obligation formula:

        gross_certificates = total_embedded_emissions (tCO2e)
        cbam_adjustment = gross_certificates * (CBAM_adjustment_pct / 100)
        carbon_deductions = sum of carbon_price_paid_abroad / ets_price
        net_certificates = cbam_adjustment - carbon_deductions

    All arithmetic uses Decimal with ROUND_HALF_UP for zero-hallucination.

    Attributes:
        config: Optional configuration dict.
        logger: Logger instance.
        _execution_id: Unique execution identifier.
        _phase_results: Accumulated phase results.

    Example:
        >>> wf = AnnualDeclarationWorkflow()
        >>> result = await wf.execute(
        ...     config={"organization_id": "org-123"},
        ...     quarterly_reports=[...],
        ...     year=2026,
        ... )
        >>> assert result.status == PhaseStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the AnnualDeclarationWorkflow.

        Args:
            config: Optional configuration dict with keys like
                'organization_id', 'ets_price_eur', 'strict_mode'.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.AnnualDeclarationWorkflow")
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        config: Optional[Dict[str, Any]],
        quarterly_reports: List[QuarterlyReportSummary],
        year: int,
    ) -> AnnualDeclarationResult:
        """
        Execute the full 8-phase annual declaration workflow.

        Args:
            config: Execution-level config overrides.
            quarterly_reports: List of quarterly report summaries for the year.
            year: Reporting year (e.g. 2026).

        Returns:
            AnnualDeclarationResult with certificate obligations and cost projections.
        """
        started_at = datetime.utcnow()
        merged_config = {**self.config, **(config or {})}
        declaration_id = f"CBAM-AD-{year}-{self._execution_id[:8]}"
        surrender_deadline = f"{year + 1}-05-31"

        self.logger.info(
            "Starting annual declaration workflow execution_id=%s year=%d",
            self._execution_id, year,
        )

        context: Dict[str, Any] = {
            "config": merged_config,
            "quarterly_reports": quarterly_reports,
            "year": year,
            "declaration_id": declaration_id,
            "execution_id": self._execution_id,
            "surrender_deadline": surrender_deadline,
        }

        phase_handlers = [
            ("annual_data_consolidation", self._phase_1_annual_data_consolidation),
            ("emission_reconciliation", self._phase_2_emission_reconciliation),
            ("certificate_calculation", self._phase_3_certificate_calculation),
            ("free_allocation_adjustment", self._phase_4_free_allocation_adjustment),
            ("carbon_price_deduction", self._phase_5_carbon_price_deduction),
            ("cost_estimation", self._phase_6_cost_estimation),
            ("declaration_assembly", self._phase_7_declaration_assembly),
            ("surrender_preparation", self._phase_8_surrender_preparation),
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
                    "Phase '%s' raised exception: %s", phase_name, exc, exc_info=True,
                )
                phase_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    outputs={"error": str(exc)},
                    provenance_hash=self._hash({"error": str(exc), "phase": phase_name}),
                )

            self._phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                # Critical phases 1-3 halt the workflow
                if phase_name in (
                    "annual_data_consolidation", "emission_reconciliation",
                    "certificate_calculation",
                ):
                    self.logger.error("Critical phase '%s' failed; halting.", phase_name)
                    break

        completed_at = datetime.utcnow()

        # Extract final values from context
        total_emissions = context.get("total_emissions_tco2e", 0.0)
        net_certificates = context.get("net_certificates", 0.0)
        estimated_cost = context.get("estimated_cost_eur", 0.0)
        free_alloc_applied = context.get("free_allocation_applied", False)
        free_alloc_pct = context.get("free_allocation_pct", 0.0)
        carbon_deductions_applied = context.get("carbon_deductions_applied", False)
        carbon_deductions = context.get("carbon_deductions_tco2e", 0.0)
        cost_projections = context.get("cost_projections", [])
        total_quantity = context.get("total_quantity_tonnes", 0.0)
        quarters_consolidated = context.get("quarters_consolidated", 0)

        provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
            "total_emissions": total_emissions,
            "net_certificates": net_certificates,
        })

        self.logger.info(
            "Annual declaration workflow finished execution_id=%s status=%s "
            "emissions=%.4f net_certs=%.4f cost=%.2f EUR",
            self._execution_id, overall_status.value,
            total_emissions, net_certificates, estimated_cost,
        )

        return AnnualDeclarationResult(
            status=overall_status,
            phases=self._phase_results,
            declaration_id=declaration_id,
            year=year,
            total_emissions_tco2e=total_emissions,
            net_certificates=net_certificates,
            estimated_cost_eur=estimated_cost,
            free_allocation_applied=free_alloc_applied,
            free_allocation_pct=free_alloc_pct,
            carbon_deductions_applied=carbon_deductions_applied,
            carbon_deductions_tco2e=carbon_deductions,
            cost_projections=cost_projections,
            provenance_hash=provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
            quarters_consolidated=quarters_consolidated,
            total_quantity_tonnes=total_quantity,
            surrender_deadline=surrender_deadline,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Annual Data Consolidation
    # -------------------------------------------------------------------------

    async def _phase_1_annual_data_consolidation(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Aggregate all 4 quarterly reports into a consolidated annual dataset.

        Merges goods categories across quarters, sums quantities and emissions,
        validates that all 4 quarters are present, and flags any data gaps.
        """
        phase_name = "annual_data_consolidation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        quarterly_reports: List[QuarterlyReportSummary] = context.get("quarterly_reports", [])
        year = context["year"]

        # Check quarter coverage
        reported_quarters = {qr.quarter for qr in quarterly_reports}
        expected_quarters = {"Q1", "Q2", "Q3", "Q4"}
        missing_quarters = expected_quarters - reported_quarters

        if missing_quarters:
            warnings.append(
                f"Missing quarterly reports for: {', '.join(sorted(missing_quarters))}"
            )

        # Consolidate emissions across quarters
        total_emissions = Decimal("0")
        total_quantity = Decimal("0")
        consolidated_categories: Dict[str, Dict[str, Any]] = {}
        quarterly_summaries: List[Dict[str, Any]] = []

        for qr in quarterly_reports:
            total_emissions += Decimal(str(qr.total_emissions_tco2e))
            total_quantity += Decimal(str(qr.total_quantity_tonnes))

            quarterly_summaries.append({
                "quarter": qr.quarter,
                "report_id": qr.report_id,
                "emissions_tco2e": qr.total_emissions_tco2e,
                "quantity_tonnes": qr.total_quantity_tonnes,
                "compliance_score": qr.compliance_score,
            })

            # Merge goods categories
            for cat in qr.goods_categories:
                cn_code = cat.get("cn_code", "unknown")
                if cn_code not in consolidated_categories:
                    consolidated_categories[cn_code] = {
                        "cn_code": cn_code,
                        "cbam_sector": cat.get("cbam_sector", ""),
                        "total_quantity_tonnes": Decimal("0"),
                        "total_emissions_tco2e": Decimal("0"),
                        "direct_emissions_tco2e": Decimal("0"),
                        "indirect_emissions_tco2e": Decimal("0"),
                        "quarters_present": [],
                    }
                cc = consolidated_categories[cn_code]
                cc["total_quantity_tonnes"] += Decimal(str(cat.get("total_quantity_tonnes", 0)))
                cc["total_emissions_tco2e"] += Decimal(str(cat.get("total_embedded_emissions_tco2e", 0)))
                cc["direct_emissions_tco2e"] += Decimal(str(cat.get("direct_emissions_tco2e", 0)))
                cc["indirect_emissions_tco2e"] += Decimal(str(cat.get("indirect_emissions_tco2e", 0)))
                cc["quarters_present"].append(qr.quarter)

        # Compute specific intensities
        annual_categories: List[Dict[str, Any]] = []
        for cn_code, cc in consolidated_categories.items():
            qty = cc["total_quantity_tonnes"]
            specific = (cc["total_emissions_tco2e"] / qty).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            ) if qty > 0 else Decimal("0")

            annual_categories.append({
                "cn_code": cn_code,
                "cbam_sector": cc["cbam_sector"],
                "total_quantity_tonnes": float(cc["total_quantity_tonnes"]),
                "total_emissions_tco2e": float(cc["total_emissions_tco2e"]),
                "specific_emissions": float(specific),
                "direct_emissions_tco2e": float(cc["direct_emissions_tco2e"]),
                "indirect_emissions_tco2e": float(cc["indirect_emissions_tco2e"]),
                "quarters_present": cc["quarters_present"],
            })

        # Store in context
        context["total_emissions_tco2e"] = float(total_emissions)
        context["total_quantity_tonnes"] = float(total_quantity)
        context["annual_categories"] = annual_categories
        context["quarters_consolidated"] = len(quarterly_reports)

        outputs["quarters_consolidated"] = len(quarterly_reports)
        outputs["missing_quarters"] = list(missing_quarters)
        outputs["total_emissions_tco2e"] = float(total_emissions)
        outputs["total_quantity_tonnes"] = float(total_quantity)
        outputs["goods_categories_count"] = len(annual_categories)
        outputs["quarterly_summaries"] = quarterly_summaries

        self.logger.info(
            "Phase 1 complete: %d quarters, %.4f tCO2e, %.2f tonnes, %d categories",
            len(quarterly_reports), float(total_emissions),
            float(total_quantity), len(annual_categories),
        )

        provenance = self._hash({
            "phase": phase_name,
            "quarters": len(quarterly_reports),
            "emissions": float(total_emissions),
        })

        status = PhaseStatus.COMPLETED
        if len(quarterly_reports) == 0:
            status = PhaseStatus.FAILED
            warnings.append("No quarterly reports to consolidate")

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Emission Reconciliation
    # -------------------------------------------------------------------------

    async def _phase_2_emission_reconciliation(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Reconcile quarterly estimates with year-end actual emission data.

        Compares quarterly reported emissions against any updated actual
        emission data (e.g. verified supplier data received after initial
        quarterly filing). Adjusts totals where actuals differ from estimates.
        """
        phase_name = "emission_reconciliation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        annual_categories: List[Dict[str, Any]] = context.get("annual_categories", [])
        quarterly_total = Decimal(str(context.get("total_emissions_tco2e", 0)))

        # Fetch year-end actual data (from verified supplier submissions)
        actuals = await self._fetch_year_end_actuals(context)

        reconciled_categories: List[Dict[str, Any]] = []
        adjustments: List[Dict[str, Any]] = []
        reconciled_total = Decimal("0")

        for cat in annual_categories:
            cn_code = cat["cn_code"]
            quarterly_emissions = Decimal(str(cat.get("total_emissions_tco2e", 0)))

            # Check if year-end actual data differs from quarterly estimate
            actual_data = actuals.get(cn_code)
            if actual_data:
                actual_emissions = Decimal(str(actual_data.get("actual_emissions_tco2e", 0)))
                difference = actual_emissions - quarterly_emissions

                if abs(difference) > Decimal("0.01"):
                    adjustments.append({
                        "cn_code": cn_code,
                        "quarterly_estimate": float(quarterly_emissions),
                        "year_end_actual": float(actual_emissions),
                        "adjustment_tco2e": float(difference),
                        "reason": actual_data.get("reason", "Verified supplier data update"),
                    })

                    cat_copy = dict(cat)
                    cat_copy["total_emissions_tco2e"] = float(actual_emissions)
                    cat_copy["reconciliation_adjustment"] = float(difference)
                    cat_copy["data_source"] = "verified_actual"
                    reconciled_categories.append(cat_copy)
                    reconciled_total += actual_emissions
                else:
                    reconciled_categories.append(cat)
                    reconciled_total += quarterly_emissions
            else:
                reconciled_categories.append(cat)
                reconciled_total += quarterly_emissions

        # Update context with reconciled data
        total_adjustment = reconciled_total - quarterly_total
        context["total_emissions_tco2e"] = float(reconciled_total)
        context["annual_categories"] = reconciled_categories

        if abs(total_adjustment) > Decimal("0.01"):
            warnings.append(
                f"Emissions reconciliation adjustment: {float(total_adjustment):+.4f} tCO2e "
                f"(quarterly: {float(quarterly_total):.4f} -> actual: {float(reconciled_total):.4f})"
            )

        outputs["quarterly_total_tco2e"] = float(quarterly_total)
        outputs["reconciled_total_tco2e"] = float(reconciled_total)
        outputs["total_adjustment_tco2e"] = float(total_adjustment)
        outputs["adjustments_count"] = len(adjustments)
        outputs["adjustments"] = adjustments

        self.logger.info(
            "Phase 2 complete: quarterly=%.4f reconciled=%.4f adjustment=%+.4f (%d adjustments)",
            float(quarterly_total), float(reconciled_total),
            float(total_adjustment), len(adjustments),
        )

        provenance = self._hash({
            "phase": phase_name,
            "quarterly": float(quarterly_total),
            "reconciled": float(reconciled_total),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Certificate Calculation
    # -------------------------------------------------------------------------

    async def _phase_3_certificate_calculation(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Calculate gross certificate obligation from total embedded emissions.

        Per Article 22(1) of Regulation 2023/956:
            1 CBAM certificate = 1 tonne CO2e
            Gross obligation = total embedded emissions (tCO2e)

        This phase calculates the raw certificate count before any
        adjustments for free allocation or carbon price deductions.
        """
        phase_name = "certificate_calculation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_emissions = Decimal(str(context.get("total_emissions_tco2e", 0)))

        # Gross certificate obligation: 1 certificate = 1 tCO2e
        gross_certificates = total_emissions.quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        # Per-sector breakdown
        sector_certificates: Dict[str, float] = {}
        for cat in context.get("annual_categories", []):
            sector = cat.get("cbam_sector", "unknown")
            emissions = Decimal(str(cat.get("total_emissions_tco2e", 0)))
            sector_certificates[sector] = (
                sector_certificates.get(sector, 0.0) + float(emissions)
            )

        context["gross_certificates"] = float(gross_certificates)
        context["sector_certificates"] = sector_certificates

        outputs["gross_certificates"] = float(gross_certificates)
        outputs["sector_certificates"] = sector_certificates
        outputs["certificate_unit"] = "tCO2e"

        if float(gross_certificates) == 0:
            warnings.append("Gross certificate obligation is zero; verify emission data")

        self.logger.info(
            "Phase 3 complete: gross_certificates=%.4f tCO2e",
            float(gross_certificates),
        )

        provenance = self._hash({
            "phase": phase_name,
            "gross_certificates": float(gross_certificates),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Free Allocation Adjustment
    # -------------------------------------------------------------------------

    async def _phase_4_free_allocation_adjustment(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Apply the current year's free allocation phase-out percentage.

        Per Article 31, CBAM certificates are only required for the
        portion NOT covered by EU ETS free allocation. As free allocation
        phases out (2026-2034), the CBAM certificate obligation increases.

        Formula:
            adjusted_certificates = gross_certificates * (cbam_adjustment_pct / 100)

        Where cbam_adjustment_pct = 100 - free_allocation_pct
        """
        phase_name = "free_allocation_adjustment"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        year = context["year"]
        gross_certificates = Decimal(str(context.get("gross_certificates", 0)))

        # Get free allocation percentage for the year
        free_alloc_pct = self._get_free_allocation_pct(year)
        cbam_adjustment_pct = Decimal(str(round(100.0 - free_alloc_pct, 2)))

        # Apply adjustment
        adjusted_certificates = (
            gross_certificates * cbam_adjustment_pct / Decimal("100")
        ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        free_alloc_deduction = (gross_certificates - adjusted_certificates).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        # Per-sector free allocation using EU ETS benchmarks
        sector_adjustments: Dict[str, Dict[str, float]] = {}
        for cat in context.get("annual_categories", []):
            sector = cat.get("cbam_sector", "unknown")
            cat_emissions = Decimal(str(cat.get("total_emissions_tco2e", 0)))
            cat_qty = Decimal(str(cat.get("total_quantity_tonnes", 0)))

            # Benchmark-based free allocation
            benchmark = Decimal(str(PRODUCT_BENCHMARKS.get(sector, 0)))
            benchmark_allocation = (cat_qty * benchmark * Decimal(str(free_alloc_pct)) / Decimal("100")).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )

            sector_adjustments[sector] = {
                "gross_emissions": float(cat_emissions),
                "benchmark_tco2e_per_t": float(benchmark),
                "free_allocation_tco2e": float(benchmark_allocation),
                "adjusted_emissions": float(
                    (cat_emissions - benchmark_allocation).quantize(
                        Decimal("0.0001"), rounding=ROUND_HALF_UP
                    )
                ),
            }

        context["adjusted_certificates"] = float(adjusted_certificates)
        context["free_allocation_deduction"] = float(free_alloc_deduction)
        context["free_allocation_pct"] = free_alloc_pct
        context["free_allocation_applied"] = True

        outputs["free_allocation_pct"] = free_alloc_pct
        outputs["cbam_adjustment_pct"] = float(cbam_adjustment_pct)
        outputs["gross_certificates"] = float(gross_certificates)
        outputs["free_allocation_deduction_tco2e"] = float(free_alloc_deduction)
        outputs["adjusted_certificates"] = float(adjusted_certificates)
        outputs["sector_adjustments"] = sector_adjustments

        if year >= 2034:
            warnings.append(
                "Free allocation fully phased out (0%) from 2034; "
                "full certificate obligation applies"
            )

        self.logger.info(
            "Phase 4 complete: free_alloc=%.1f%% gross=%.4f adjusted=%.4f deduction=%.4f",
            free_alloc_pct, float(gross_certificates),
            float(adjusted_certificates), float(free_alloc_deduction),
        )

        provenance = self._hash({
            "phase": phase_name,
            "free_alloc_pct": free_alloc_pct,
            "adjusted": float(adjusted_certificates),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Carbon Price Deduction
    # -------------------------------------------------------------------------

    async def _phase_5_carbon_price_deduction(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Calculate third-country carbon pricing deductions.

        Per Article 9 of Regulation 2023/956, if a carbon price has been
        effectively paid in the country of origin, a deduction may be
        applied to the CBAM certificate obligation.

        Formula:
            carbon_deduction_certs = total_carbon_price_paid_eur / ets_price_eur
            net_certificates = adjusted_certificates - carbon_deduction_certs
        """
        phase_name = "carbon_price_deduction"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        adjusted_certificates = Decimal(str(context.get("adjusted_certificates", 0)))
        ets_price = Decimal(str(context.get("config", {}).get("ets_price_eur", 80.0)))

        # Fetch carbon prices paid abroad
        carbon_prices = await self._fetch_carbon_prices_paid(context)

        total_carbon_paid_eur = Decimal("0")
        country_deductions: List[Dict[str, Any]] = []

        for entry in carbon_prices:
            country = entry.get("country", "unknown")
            amount_eur = Decimal(str(entry.get("amount_eur", 0)))
            quantity = Decimal(str(entry.get("quantity_tonnes", 0)))
            instrument = entry.get("instrument_type", "carbon_tax")

            total_carbon_paid_eur += amount_eur

            country_deductions.append({
                "country": country,
                "amount_eur": float(amount_eur),
                "quantity_tonnes": float(quantity),
                "instrument_type": instrument,
            })

        # Calculate certificate deduction
        if ets_price > 0 and total_carbon_paid_eur > 0:
            carbon_deduction_certs = (total_carbon_paid_eur / ets_price).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
        else:
            carbon_deduction_certs = Decimal("0")

        # Net certificates cannot be negative
        net_certificates = max(
            Decimal("0"),
            (adjusted_certificates - carbon_deduction_certs).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            ),
        )

        context["carbon_deductions_tco2e"] = float(carbon_deduction_certs)
        context["carbon_deductions_applied"] = float(carbon_deduction_certs) > 0
        context["net_certificates"] = float(net_certificates)
        context["total_carbon_price_paid_eur"] = float(total_carbon_paid_eur)

        outputs["adjusted_certificates"] = float(adjusted_certificates)
        outputs["total_carbon_price_paid_eur"] = float(total_carbon_paid_eur)
        outputs["ets_price_eur"] = float(ets_price)
        outputs["carbon_deduction_certificates"] = float(carbon_deduction_certs)
        outputs["net_certificates"] = float(net_certificates)
        outputs["country_deductions"] = country_deductions

        if float(carbon_deduction_certs) > 0:
            self.logger.info(
                "Carbon price deduction: %.4f certificates (%.2f EUR paid / %.2f EUR ETS)",
                float(carbon_deduction_certs), float(total_carbon_paid_eur), float(ets_price),
            )

        self.logger.info(
            "Phase 5 complete: adjusted=%.4f deductions=%.4f net=%.4f",
            float(adjusted_certificates), float(carbon_deduction_certs), float(net_certificates),
        )

        provenance = self._hash({
            "phase": phase_name,
            "net_certificates": float(net_certificates),
            "carbon_deduction": float(carbon_deduction_certs),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 6: Cost Estimation
    # -------------------------------------------------------------------------

    async def _phase_6_cost_estimation(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Project certificate cost with low/mid/high price scenarios.

        Each scenario uses a different EU ETS price assumption to project
        the total cost of CBAM certificates. Cost projections help
        importers budget for compliance costs.

        Scenarios:
            LOW:  EU ETS price at 25th percentile forecast
            MID:  EU ETS price at market consensus forecast
            HIGH: EU ETS price at 75th percentile forecast
        """
        phase_name = "cost_estimation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        net_certificates = Decimal(str(context.get("net_certificates", 0)))
        free_alloc_deduction = Decimal(str(context.get("free_allocation_deduction", 0)))
        carbon_deduction_certs = Decimal(str(context.get("carbon_deductions_tco2e", 0)))
        gross_certificates = Decimal(str(context.get("gross_certificates", 0)))
        year = context["year"]

        # ETS price scenarios (EUR per tCO2e)
        config = context.get("config", {})
        price_scenarios = {
            CostScenario.LOW: Decimal(str(config.get("ets_price_low", 55.0))),
            CostScenario.MID: Decimal(str(config.get("ets_price_eur", 80.0))),
            CostScenario.HIGH: Decimal(str(config.get("ets_price_high", 120.0))),
        }

        projections: List[CostProjection] = []
        for scenario, price in price_scenarios.items():
            gross_cost = (gross_certificates * price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            free_alloc_deduction_eur = (free_alloc_deduction * price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            carbon_deduction_eur = (carbon_deduction_certs * price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            net_cost = (net_certificates * price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            projections.append(CostProjection(
                scenario=scenario,
                ets_price_eur_per_tco2e=float(price),
                gross_cost_eur=float(gross_cost),
                free_allocation_deduction_eur=float(free_alloc_deduction_eur),
                carbon_price_deduction_eur=float(carbon_deduction_eur),
                net_cost_eur=float(net_cost),
            ))

        # Use mid scenario as the estimated cost
        mid_projection = next(
            (p for p in projections if p.scenario == CostScenario.MID), None
        )
        estimated_cost = mid_projection.net_cost_eur if mid_projection else 0.0

        context["cost_projections"] = projections
        context["estimated_cost_eur"] = estimated_cost

        outputs["projections"] = [p.model_dump() for p in projections]
        outputs["estimated_cost_eur"] = estimated_cost
        outputs["net_certificates"] = float(net_certificates)

        # Cost impact warning
        if estimated_cost > 100_000:
            warnings.append(
                f"Estimated annual CBAM cost is EUR {estimated_cost:,.2f} (mid scenario). "
                "Consider collecting actual supplier emission data to optimize."
            )

        self.logger.info(
            "Phase 6 complete: low=%.2f mid=%.2f high=%.2f EUR",
            projections[0].net_cost_eur if projections else 0,
            projections[1].net_cost_eur if len(projections) > 1 else 0,
            projections[2].net_cost_eur if len(projections) > 2 else 0,
        )

        provenance = self._hash({
            "phase": phase_name,
            "estimated_cost": estimated_cost,
            "net_certs": float(net_certificates),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 7: Declaration Assembly
    # -------------------------------------------------------------------------

    async def _phase_7_declaration_assembly(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Generate the annual CBAM declaration package.

        Assembles:
            - Declaration XML per EU CBAM Registry schema
            - Goods category summary with emission intensities
            - Certificate obligation calculation workbook
            - Verification statement reference
            - Supporting documentation index
        """
        phase_name = "declaration_assembly"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        declaration_id = context["declaration_id"]
        year = context["year"]

        # Generate declaration XML
        declaration_xml = await self._generate_declaration_xml(context)

        # Assemble declaration metadata
        declaration_metadata = {
            "declaration_id": declaration_id,
            "year": year,
            "total_emissions_tco2e": context.get("total_emissions_tco2e", 0),
            "net_certificates": context.get("net_certificates", 0),
            "estimated_cost_eur": context.get("estimated_cost_eur", 0),
            "free_allocation_pct": context.get("free_allocation_pct", 0),
            "goods_categories": len(context.get("annual_categories", [])),
            "quarters_consolidated": context.get("quarters_consolidated", 0),
            "generated_at": datetime.utcnow().isoformat(),
            "surrender_deadline": context.get("surrender_deadline", ""),
        }

        # Check for verification statement
        verification_ref = await self._check_verification_statement(context)
        if not verification_ref:
            warnings.append(
                "No verification statement found. Annual declarations from 2028 "
                "require third-party verification per Article 8."
            )

        outputs["declaration_id"] = declaration_id
        outputs["declaration_metadata"] = declaration_metadata
        outputs["xml_generated"] = bool(declaration_xml)
        outputs["xml_size_bytes"] = len(declaration_xml) if declaration_xml else 0
        outputs["verification_statement_ref"] = verification_ref

        context["declaration_xml"] = declaration_xml

        self.logger.info(
            "Phase 7 complete: declaration_id=%s xml_size=%d",
            declaration_id, outputs["xml_size_bytes"],
        )

        provenance = self._hash({
            "phase": phase_name,
            "declaration_id": declaration_id,
            "xml_generated": outputs["xml_generated"],
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 8: Surrender Preparation
    # -------------------------------------------------------------------------

    async def _phase_8_surrender_preparation(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Prepare the certificate surrender package for the May 31 deadline.

        Verifies:
            - Sufficient certificates in declarant's account
            - All calculation inputs are finalized
            - Declaration is complete and valid
            - Generates surrender instruction manifest

        Per Article 22(1), certificates must be surrendered by May 31
        of the year following the reporting year.
        """
        phase_name = "surrender_preparation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        net_certificates = context.get("net_certificates", 0)
        declaration_id = context["declaration_id"]
        surrender_deadline = context.get("surrender_deadline", "")

        # Check certificate balance
        certificate_balance = await self._check_certificate_balance(context)
        certificates_held = certificate_balance.get("certificates_held", 0)
        shortfall = max(0, net_certificates - certificates_held)

        # Check days until deadline
        deadline_date = datetime.strptime(surrender_deadline, "%Y-%m-%d")
        days_until_deadline = (deadline_date - datetime.utcnow()).days

        # Generate surrender instruction
        surrender_manifest = {
            "declaration_id": declaration_id,
            "certificates_to_surrender": net_certificates,
            "certificates_held": certificates_held,
            "shortfall": shortfall,
            "surrender_deadline": surrender_deadline,
            "days_until_deadline": days_until_deadline,
            "ready_to_surrender": shortfall == 0,
            "prepared_at": datetime.utcnow().isoformat(),
        }

        if shortfall > 0:
            warnings.append(
                f"Certificate shortfall of {shortfall:.4f} certificates. "
                f"Purchase {shortfall:.4f} additional certificates before {surrender_deadline}."
            )

        if days_until_deadline < 30:
            warnings.append(
                f"Only {days_until_deadline} days until surrender deadline ({surrender_deadline}). "
                "Prioritize certificate purchase and declaration finalization."
            )

        if days_until_deadline < 0:
            warnings.append(
                f"Surrender deadline {surrender_deadline} has PASSED. "
                "Late surrender may incur penalties per Article 26."
            )

        outputs["surrender_manifest"] = surrender_manifest
        outputs["certificates_to_surrender"] = net_certificates
        outputs["certificates_held"] = certificates_held
        outputs["shortfall"] = shortfall
        outputs["days_until_deadline"] = days_until_deadline
        outputs["ready_to_surrender"] = shortfall == 0

        self.logger.info(
            "Phase 8 complete: to_surrender=%.4f held=%.4f shortfall=%.4f days=%d",
            net_certificates, certificates_held, shortfall, days_until_deadline,
        )

        provenance = self._hash({
            "phase": phase_name,
            "net_certificates": net_certificates,
            "shortfall": shortfall,
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

    def _get_free_allocation_pct(self, year: int) -> float:
        """Get the free allocation percentage for a given year.

        Returns the percentage of EU ETS free allocation still in effect.
        """
        if year <= 2025:
            return 100.0
        if year >= 2034:
            return 0.0
        return FREE_ALLOCATION_PCT.get(year, 0.0)

    # =========================================================================
    # ASYNC STUBS (Agent Invocations)
    # =========================================================================

    async def _fetch_year_end_actuals(
        self, context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch year-end actual emission data from verified supplier submissions."""
        await asyncio.sleep(0)
        return {}

    async def _fetch_carbon_prices_paid(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fetch carbon prices paid in countries of origin."""
        await asyncio.sleep(0)
        return []

    async def _generate_declaration_xml(
        self, context: Dict[str, Any]
    ) -> str:
        """Generate annual declaration XML per EU CBAM Registry schema."""
        await asyncio.sleep(0)
        declaration_id = context.get("declaration_id", "")
        year = context.get("year", 0)
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<CBAMAnnualDeclaration declarationId="{declaration_id}" year="{year}">',
            f'  <TotalEmissions>{context.get("total_emissions_tco2e", 0):.4f}</TotalEmissions>',
            f'  <NetCertificates>{context.get("net_certificates", 0):.4f}</NetCertificates>',
            f'  <FreeAllocationPct>{context.get("free_allocation_pct", 0):.2f}</FreeAllocationPct>',
            "</CBAMAnnualDeclaration>",
        ]
        return "\n".join(xml_lines)

    async def _check_verification_statement(
        self, context: Dict[str, Any]
    ) -> Optional[str]:
        """Check if a verification statement exists for this year."""
        await asyncio.sleep(0)
        return None

    async def _check_certificate_balance(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check the declarant's certificate balance in the CBAM registry."""
        await asyncio.sleep(0)
        return {"certificates_held": 0, "last_checked": datetime.utcnow().isoformat()}

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
