# -*- coding: utf-8 -*-
"""
Tenant Engagement Workflow
===============================

3-phase workflow for tenant energy engagement within PACK-032
Building Energy Assessment Pack.

Phases:
    1. TenantProfiling     -- Space allocation, metering, lease type
    2. Benchmarking        -- Tenant vs building vs portfolio comparison
    3. EngagementReport    -- Tenant-facing energy report, green lease compliance

Zero-hallucination: all benchmarking uses deterministic normalisation and
validated reference data. No LLM calls in the calculation path.

Schedule: quarterly
Estimated duration: 60 minutes per tenant

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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


class LeaseType(str, Enum):
    """Lease arrangement types."""

    GROSS = "gross"
    NET = "net"
    MODIFIED_GROSS = "modified_gross"
    TRIPLE_NET = "triple_net"
    GREEN = "green"


class MeteringType(str, Enum):
    """Metering arrangement types."""

    DIRECT = "direct"
    SUB_METERED = "sub_metered"
    APPORTIONED = "apportioned"
    NONE = "none"


class TenantRating(str, Enum):
    """Tenant performance rating."""

    EXEMPLARY = "exemplary"
    GOOD = "good"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    POOR = "poor"


class GreenLeaseClauseStatus(str, Enum):
    """Green lease clause compliance status."""

    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


# =============================================================================
# ZERO-HALLUCINATION REFERENCE CONSTANTS
# =============================================================================

# EUI benchmarks by tenant use type (kWh/m2/yr) -- CIBSE TM46 / BBP REEB
TENANT_EUI_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "general_office": {"best": 95.0, "good": 128.0, "typical": 230.0, "poor": 350.0},
    "trading_floor": {"best": 250.0, "good": 350.0, "typical": 500.0, "poor": 700.0},
    "server_room": {"best": 400.0, "good": 600.0, "typical": 800.0, "poor": 1200.0},
    "retail_unit": {"best": 140.0, "good": 190.0, "typical": 305.0, "poor": 450.0},
    "restaurant": {"best": 240.0, "good": 320.0, "typical": 450.0, "poor": 650.0},
    "gym_fitness": {"best": 200.0, "good": 280.0, "typical": 400.0, "poor": 550.0},
    "common_areas": {"best": 60.0, "good": 90.0, "typical": 140.0, "poor": 200.0},
    "storage": {"best": 20.0, "good": 40.0, "typical": 70.0, "poor": 120.0},
}

# CO2 emission factor (kgCO2/kWh) -- DEFRA 2024
ELECTRICITY_EF: float = 0.207

# Green lease standard clauses -- BBP Green Lease Toolkit
GREEN_LEASE_CLAUSES: List[Dict[str, str]] = [
    {"id": "GL01", "name": "Energy data sharing", "description": "Tenant shares energy consumption data quarterly."},
    {"id": "GL02", "name": "No alterations restriction", "description": "Tenant must not downgrade energy efficiency of fit-out."},
    {"id": "GL03", "name": "Metering cooperation", "description": "Tenant cooperates with sub-metering installation."},
    {"id": "GL04", "name": "Sustainability targets", "description": "Tenant commits to building energy reduction targets."},
    {"id": "GL05", "name": "Waste management", "description": "Tenant participates in building recycling programme."},
    {"id": "GL06", "name": "Fit-out standards", "description": "Tenant fit-out meets minimum energy efficiency standards."},
    {"id": "GL07", "name": "EPC minimum rating", "description": "Tenant space must achieve minimum EPC rating."},
    {"id": "GL08", "name": "Renewable energy participation", "description": "Tenant supports on-site renewable energy."},
]

# Occupancy density defaults (m2 per person) by use type
OCCUPANCY_DENSITY: Dict[str, float] = {
    "general_office": 10.0,
    "trading_floor": 6.0,
    "server_room": 50.0,
    "retail_unit": 15.0,
    "restaurant": 5.0,
    "gym_fitness": 8.0,
    "common_areas": 20.0,
    "storage": 100.0,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class TenantSpace(BaseModel):
    """Tenant space allocation record."""

    tenant_id: str = Field(default_factory=lambda: f"ten-{uuid.uuid4().hex[:8]}")
    tenant_name: str = Field(default="")
    floor: str = Field(default="", description="Floor level (e.g. 'GF', '1', '2')")
    zone: str = Field(default="", description="Zone within floor")
    use_type: str = Field(default="general_office")
    area_sqm: float = Field(default=0.0, ge=0.0)
    lease_type: LeaseType = Field(default=LeaseType.GROSS)
    metering: MeteringType = Field(default=MeteringType.APPORTIONED)
    lease_start_date: str = Field(default="")
    lease_end_date: str = Field(default="")
    green_lease: bool = Field(default=False)
    occupants: int = Field(default=0, ge=0)
    operating_hours_per_day: float = Field(default=10.0, ge=0.0, le=24.0)


class TenantEnergyData(BaseModel):
    """Energy consumption data for a tenant space."""

    tenant_id: str = Field(default="")
    period: str = Field(default="", description="YYYY-MM or YYYY-Q1")
    electricity_kwh: float = Field(default=0.0, ge=0.0)
    gas_kwh: float = Field(default=0.0, ge=0.0)
    total_kwh: float = Field(default=0.0, ge=0.0)
    cost_eur: float = Field(default=0.0, ge=0.0)
    data_source: str = Field(default="sub_meter", description="sub_meter|apportioned|estimated")


class BuildingTotalData(BaseModel):
    """Whole-building energy data for comparison."""

    total_floor_area_sqm: float = Field(default=0.0, ge=0.0)
    annual_electricity_kwh: float = Field(default=0.0, ge=0.0)
    annual_gas_kwh: float = Field(default=0.0, ge=0.0)
    annual_total_kwh: float = Field(default=0.0, ge=0.0)
    annual_cost_eur: float = Field(default=0.0, ge=0.0)
    eui_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    co2_kg_per_sqm: float = Field(default=0.0, ge=0.0)
    epc_band: str = Field(default="")


class PortfolioData(BaseModel):
    """Portfolio average data for comparison."""

    portfolio_name: str = Field(default="")
    avg_eui_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    avg_co2_kg_per_sqm: float = Field(default=0.0, ge=0.0)
    best_eui_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    worst_eui_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    building_count: int = Field(default=0, ge=0)


class TenantBenchmark(BaseModel):
    """Tenant benchmarking result."""

    tenant_id: str = Field(default="")
    tenant_name: str = Field(default="")
    eui_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    co2_kg_per_sqm: float = Field(default=0.0, ge=0.0)
    cost_eur_per_sqm: float = Field(default=0.0, ge=0.0)
    benchmark_category: str = Field(default="", description="best|good|typical|poor")
    vs_building_pct: float = Field(default=0.0, description="% above/below building avg")
    vs_portfolio_pct: float = Field(default=0.0, description="% above/below portfolio avg")
    vs_benchmark_pct: float = Field(default=0.0, description="% above/below use-type benchmark")
    rating: TenantRating = Field(default=TenantRating.AVERAGE)
    rank_in_building: int = Field(default=0, ge=0)
    total_tenants: int = Field(default=0, ge=0)


class GreenLeaseAssessment(BaseModel):
    """Green lease clause compliance assessment."""

    clause_id: str = Field(default="")
    clause_name: str = Field(default="")
    status: GreenLeaseClauseStatus = Field(default=GreenLeaseClauseStatus.NOT_APPLICABLE)
    notes: str = Field(default="")


class TenantReport(BaseModel):
    """Tenant-facing energy report."""

    tenant_id: str = Field(default="")
    tenant_name: str = Field(default="")
    reporting_period: str = Field(default="")
    total_consumption_kwh: float = Field(default=0.0, ge=0.0)
    total_cost_eur: float = Field(default=0.0, ge=0.0)
    eui_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    co2_emissions_kg: float = Field(default=0.0, ge=0.0)
    benchmark_rating: str = Field(default="")
    performance_trend: str = Field(default="", description="improving|stable|deteriorating")
    recommendations: List[str] = Field(default_factory=list)
    green_lease_clauses: List[GreenLeaseAssessment] = Field(default_factory=list)
    green_lease_compliance_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class TenantEngagementInput(BaseModel):
    """Input data model for TenantEngagementWorkflow."""

    building_name: str = Field(default="")
    tenants: List[TenantSpace] = Field(default_factory=list)
    tenant_energy_data: List[TenantEnergyData] = Field(default_factory=list)
    building_data: BuildingTotalData = Field(default_factory=BuildingTotalData)
    portfolio_data: Optional[PortfolioData] = None
    reporting_period: str = Field(default="", description="e.g. 2025-Q1 or 2025")
    previous_period_data: List[TenantEnergyData] = Field(default_factory=list)
    entity_id: str = Field(default="")
    tenant_context_id: str = Field(default="")

    @field_validator("tenants")
    @classmethod
    def validate_tenants(cls, v: List[TenantSpace]) -> List[TenantSpace]:
        """At least one tenant must be provided."""
        if not v:
            raise ValueError("At least one tenant space must be provided")
        return v


class TenantEngagementResult(BaseModel):
    """Complete result from tenant engagement workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="tenant_engagement")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    building_name: str = Field(default="")
    reporting_period: str = Field(default="")
    tenants_assessed: int = Field(default=0)
    benchmarks: List[TenantBenchmark] = Field(default_factory=list)
    tenant_reports: List[TenantReport] = Field(default_factory=list)
    best_performing_tenant: str = Field(default="")
    worst_performing_tenant: str = Field(default="")
    building_eui: float = Field(default=0.0, ge=0.0)
    average_tenant_eui: float = Field(default=0.0, ge=0.0)
    green_lease_compliance_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class TenantEngagementWorkflow:
    """
    3-phase tenant energy engagement workflow.

    Profiles tenant spaces and metering, benchmarks each tenant against
    building average, portfolio data, and use-type references, then
    generates tenant-facing reports with green lease compliance.

    Zero-hallucination: all benchmarking uses validated EUI reference data
    (CIBSE TM46, BBP REEB) and deterministic normalisation.

    Example:
        >>> wf = TenantEngagementWorkflow()
        >>> inp = TenantEngagementInput(
        ...     tenants=[TenantSpace(tenant_name="Acme", area_sqm=500)],
        ...     tenant_energy_data=[TenantEnergyData(tenant_id="...", total_kwh=50000)]
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize TenantEngagementWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._benchmarks: List[TenantBenchmark] = []
        self._reports: List[TenantReport] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[TenantEngagementInput] = None,
    ) -> TenantEngagementResult:
        """Execute the 3-phase tenant engagement workflow."""
        if input_data is None:
            raise ValueError("input_data must be provided")

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting tenant engagement workflow %s for %s, %d tenants",
            self.workflow_id, input_data.building_name, len(input_data.tenants),
        )

        self._phase_results = []
        self._benchmarks = []
        self._reports = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_tenant_profiling(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_benchmarking(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_engagement_report(input_data)
            self._phase_results.append(phase3)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Tenant engagement workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        best_tenant = min(self._benchmarks, key=lambda b: b.eui_kwh_per_sqm).tenant_name if self._benchmarks else ""
        worst_tenant = max(self._benchmarks, key=lambda b: b.eui_kwh_per_sqm).tenant_name if self._benchmarks else ""
        avg_eui = sum(b.eui_kwh_per_sqm for b in self._benchmarks) / max(len(self._benchmarks), 1)
        gl_compliance = sum(r.green_lease_compliance_pct for r in self._reports) / max(len(self._reports), 1)

        result = TenantEngagementResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            building_name=input_data.building_name,
            reporting_period=input_data.reporting_period,
            tenants_assessed=len(self._benchmarks),
            benchmarks=self._benchmarks,
            tenant_reports=self._reports,
            best_performing_tenant=best_tenant,
            worst_performing_tenant=worst_tenant,
            building_eui=round(input_data.building_data.eui_kwh_per_sqm, 2),
            average_tenant_eui=round(avg_eui, 2),
            green_lease_compliance_pct=round(gl_compliance, 1),
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Tenant engagement workflow %s completed in %.2fs: %d tenants, "
            "avg EUI=%.1f kWh/m2, best=%s",
            self.workflow_id, elapsed, len(self._benchmarks), avg_eui, best_tenant,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Tenant Profiling
    # -------------------------------------------------------------------------

    async def _phase_tenant_profiling(
        self, input_data: TenantEngagementInput
    ) -> PhaseResult:
        """Profile tenant spaces: allocation, metering, lease type."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_tenant_area = sum(t.area_sqm for t in input_data.tenants)
        building_area = input_data.building_data.total_floor_area_sqm
        if building_area > 0 and total_tenant_area > building_area * 1.1:
            warnings.append(
                f"Total tenant area ({total_tenant_area:.0f} m2) exceeds "
                f"building area ({building_area:.0f} m2)"
            )

        metering_summary: Dict[str, int] = {}
        lease_summary: Dict[str, int] = {}
        green_lease_count = 0

        for tenant in input_data.tenants:
            mt = tenant.metering.value
            metering_summary[mt] = metering_summary.get(mt, 0) + 1
            lt = tenant.lease_type.value
            lease_summary[lt] = lease_summary.get(lt, 0) + 1
            if tenant.green_lease:
                green_lease_count += 1
            if tenant.area_sqm <= 0:
                warnings.append(f"Tenant {tenant.tenant_name} has zero area")
            if tenant.metering == MeteringType.NONE:
                warnings.append(f"Tenant {tenant.tenant_name} has no metering")

        outputs["tenant_count"] = len(input_data.tenants)
        outputs["total_tenant_area_sqm"] = round(total_tenant_area, 2)
        outputs["building_area_sqm"] = round(building_area, 2)
        outputs["area_coverage_pct"] = round(
            total_tenant_area / max(building_area, 1) * 100, 1
        )
        outputs["metering_summary"] = metering_summary
        outputs["lease_summary"] = lease_summary
        outputs["green_lease_count"] = green_lease_count
        outputs["green_lease_pct"] = round(
            green_lease_count / max(len(input_data.tenants), 1) * 100, 1
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 TenantProfiling: %d tenants, %.0f m2, %d green leases",
            len(input_data.tenants), total_tenant_area, green_lease_count,
        )
        return PhaseResult(
            phase_name="tenant_profiling", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Benchmarking
    # -------------------------------------------------------------------------

    async def _phase_benchmarking(
        self, input_data: TenantEngagementInput
    ) -> PhaseResult:
        """Benchmark tenants against building, portfolio, and reference data."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        building_eui = input_data.building_data.eui_kwh_per_sqm
        portfolio_eui = input_data.portfolio_data.avg_eui_kwh_per_sqm if input_data.portfolio_data else 0.0

        # Aggregate energy by tenant
        tenant_energy: Dict[str, float] = {}
        tenant_cost: Dict[str, float] = {}
        for ted in input_data.tenant_energy_data:
            tid = ted.tenant_id
            tenant_energy[tid] = tenant_energy.get(tid, 0.0) + ted.total_kwh
            tenant_cost[tid] = tenant_cost.get(tid, 0.0) + ted.cost_eur

        for tenant in input_data.tenants:
            tid = tenant.tenant_id
            area = tenant.area_sqm if tenant.area_sqm > 0 else 1.0
            total_kwh = tenant_energy.get(tid, 0.0)
            total_cost = tenant_cost.get(tid, 0.0)
            eui = total_kwh / area if area > 0 else 0.0
            co2 = total_kwh * ELECTRICITY_EF / area if area > 0 else 0.0
            cost_sqm = total_cost / area if area > 0 else 0.0

            # Benchmark category from reference data
            benchmarks = TENANT_EUI_BENCHMARKS.get(tenant.use_type, TENANT_EUI_BENCHMARKS["general_office"])
            if eui <= benchmarks["best"]:
                bench_cat = "best"
                rating = TenantRating.EXEMPLARY
            elif eui <= benchmarks["good"]:
                bench_cat = "good"
                rating = TenantRating.GOOD
            elif eui <= benchmarks["typical"]:
                bench_cat = "typical"
                rating = TenantRating.AVERAGE
            elif eui <= benchmarks["poor"]:
                bench_cat = "below_average"
                rating = TenantRating.BELOW_AVERAGE
            else:
                bench_cat = "poor"
                rating = TenantRating.POOR

            vs_building = ((eui - building_eui) / building_eui * 100) if building_eui > 0 else 0.0
            vs_portfolio = ((eui - portfolio_eui) / portfolio_eui * 100) if portfolio_eui > 0 else 0.0
            typical_ref = benchmarks["typical"]
            vs_benchmark = ((eui - typical_ref) / typical_ref * 100) if typical_ref > 0 else 0.0

            self._benchmarks.append(TenantBenchmark(
                tenant_id=tid,
                tenant_name=tenant.tenant_name,
                eui_kwh_per_sqm=round(eui, 2),
                co2_kg_per_sqm=round(co2, 2),
                cost_eur_per_sqm=round(cost_sqm, 2),
                benchmark_category=bench_cat,
                vs_building_pct=round(vs_building, 1),
                vs_portfolio_pct=round(vs_portfolio, 1),
                vs_benchmark_pct=round(vs_benchmark, 1),
                rating=rating,
                total_tenants=len(input_data.tenants),
            ))

        # Rank tenants
        self._benchmarks.sort(key=lambda b: b.eui_kwh_per_sqm)
        for i, bm in enumerate(self._benchmarks, 1):
            bm.rank_in_building = i

        outputs["tenants_benchmarked"] = len(self._benchmarks)
        outputs["avg_eui"] = round(
            sum(b.eui_kwh_per_sqm for b in self._benchmarks) / max(len(self._benchmarks), 1), 2
        )
        outputs["best_eui"] = round(self._benchmarks[0].eui_kwh_per_sqm, 2) if self._benchmarks else 0.0
        outputs["worst_eui"] = round(self._benchmarks[-1].eui_kwh_per_sqm, 2) if self._benchmarks else 0.0
        outputs["rating_distribution"] = self._rating_distribution()

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 Benchmarking: %d tenants, avg EUI=%.1f, best=%.1f, worst=%.1f",
            len(self._benchmarks), outputs["avg_eui"],
            outputs["best_eui"], outputs["worst_eui"],
        )
        return PhaseResult(
            phase_name="benchmarking", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Engagement Report
    # -------------------------------------------------------------------------

    async def _phase_engagement_report(
        self, input_data: TenantEngagementInput
    ) -> PhaseResult:
        """Generate tenant-facing energy reports with green lease compliance."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Previous period data for trend
        prev_energy: Dict[str, float] = {}
        for ted in input_data.previous_period_data:
            prev_energy[ted.tenant_id] = prev_energy.get(ted.tenant_id, 0.0) + ted.total_kwh

        for bm in self._benchmarks:
            tenant = next((t for t in input_data.tenants if t.tenant_id == bm.tenant_id), None)
            if tenant is None:
                continue

            total_kwh = bm.eui_kwh_per_sqm * tenant.area_sqm
            total_cost = bm.cost_eur_per_sqm * tenant.area_sqm
            co2 = total_kwh * ELECTRICITY_EF

            # Trend analysis
            prev_kwh = prev_energy.get(bm.tenant_id, 0.0)
            if prev_kwh > 0:
                change = (total_kwh - prev_kwh) / prev_kwh
                if change < -0.02:
                    trend = "improving"
                elif change > 0.02:
                    trend = "deteriorating"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            # Generate recommendations
            recommendations = self._generate_tenant_recommendations(bm, tenant)

            # Green lease compliance
            gl_clauses: List[GreenLeaseAssessment] = []
            if tenant.green_lease:
                gl_clauses = self._assess_green_lease(tenant, bm)

            compliant = sum(1 for c in gl_clauses if c.status == GreenLeaseClauseStatus.COMPLIANT)
            applicable = sum(1 for c in gl_clauses if c.status != GreenLeaseClauseStatus.NOT_APPLICABLE)
            gl_pct = (compliant / max(applicable, 1)) * 100

            self._reports.append(TenantReport(
                tenant_id=bm.tenant_id,
                tenant_name=bm.tenant_name,
                reporting_period=input_data.reporting_period,
                total_consumption_kwh=round(total_kwh, 2),
                total_cost_eur=round(total_cost, 2),
                eui_kwh_per_sqm=round(bm.eui_kwh_per_sqm, 2),
                co2_emissions_kg=round(co2, 2),
                benchmark_rating=bm.rating.value,
                performance_trend=trend,
                recommendations=recommendations,
                green_lease_clauses=gl_clauses,
                green_lease_compliance_pct=round(gl_pct, 1),
            ))

        total_reports = len(self._reports)
        avg_compliance = sum(r.green_lease_compliance_pct for r in self._reports) / max(total_reports, 1)

        outputs["reports_generated"] = total_reports
        outputs["avg_green_lease_compliance_pct"] = round(avg_compliance, 1)
        outputs["tenants_improving"] = sum(1 for r in self._reports if r.performance_trend == "improving")
        outputs["tenants_deteriorating"] = sum(1 for r in self._reports if r.performance_trend == "deteriorating")
        outputs["tenants_stable"] = sum(1 for r in self._reports if r.performance_trend == "stable")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 EngagementReport: %d reports, GL compliance=%.1f%%",
            total_reports, avg_compliance,
        )
        return PhaseResult(
            phase_name="engagement_report", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_tenant_recommendations(
        self, benchmark: TenantBenchmark, tenant: TenantSpace
    ) -> List[str]:
        """Generate recommendations for a tenant."""
        recs: List[str] = []
        if benchmark.rating in (TenantRating.BELOW_AVERAGE, TenantRating.POOR):
            recs.append("Review out-of-hours energy use and implement switch-off campaigns.")
            recs.append("Consider LED lighting throughout your space.")
            recs.append("Install plug-load management for desk equipment.")
        if benchmark.rating == TenantRating.AVERAGE:
            recs.append("Investigate sub-metering to identify the largest consumers.")
            recs.append("Review HVAC setpoints with your facilities manager.")
        if tenant.metering == MeteringType.APPORTIONED:
            recs.append("Request sub-metering to track actual consumption instead of apportioned data.")
        if not tenant.green_lease:
            recs.append("Consider adopting a green lease to align landlord-tenant sustainability goals.")
        if benchmark.vs_building_pct > 20:
            recs.append(
                f"Your EUI is {benchmark.vs_building_pct:.0f}% above the building average. "
                "Conduct a focused energy audit of your space."
            )
        if benchmark.rating == TenantRating.EXEMPLARY:
            recs.append("Congratulations on exemplary performance. Consider sharing best practices.")
        return recs

    def _assess_green_lease(
        self, tenant: TenantSpace, benchmark: TenantBenchmark
    ) -> List[GreenLeaseAssessment]:
        """Assess green lease clause compliance."""
        assessments: List[GreenLeaseAssessment] = []
        for clause in GREEN_LEASE_CLAUSES:
            status = GreenLeaseClauseStatus.NOT_APPLICABLE
            notes = ""

            if clause["id"] == "GL01":
                status = (
                    GreenLeaseClauseStatus.COMPLIANT
                    if tenant.metering in (MeteringType.DIRECT, MeteringType.SUB_METERED)
                    else GreenLeaseClauseStatus.NON_COMPLIANT
                )
                notes = "Energy data sharing via sub-metering" if status == GreenLeaseClauseStatus.COMPLIANT else "No metering data available"
            elif clause["id"] == "GL03":
                status = (
                    GreenLeaseClauseStatus.COMPLIANT
                    if tenant.metering != MeteringType.NONE
                    else GreenLeaseClauseStatus.NON_COMPLIANT
                )
            elif clause["id"] == "GL04":
                if benchmark.rating in (TenantRating.EXEMPLARY, TenantRating.GOOD):
                    status = GreenLeaseClauseStatus.COMPLIANT
                    notes = "Meeting sustainability targets"
                elif benchmark.rating == TenantRating.AVERAGE:
                    status = GreenLeaseClauseStatus.PARTIAL
                    notes = "Near target but improvement needed"
                else:
                    status = GreenLeaseClauseStatus.NON_COMPLIANT
                    notes = "Below sustainability targets"
            elif clause["id"] == "GL05":
                status = GreenLeaseClauseStatus.COMPLIANT  # Default compliant
                notes = "Participating in building waste programme"
            elif clause["id"] in ("GL02", "GL06", "GL07", "GL08"):
                status = GreenLeaseClauseStatus.COMPLIANT  # Assumed compliant without evidence
                notes = "Assumed compliant pending verification"

            assessments.append(GreenLeaseAssessment(
                clause_id=clause["id"],
                clause_name=clause["name"],
                status=status,
                notes=notes,
            ))
        return assessments

    def _rating_distribution(self) -> Dict[str, int]:
        """Count tenants by rating."""
        dist: Dict[str, int] = {}
        for bm in self._benchmarks:
            dist[bm.rating.value] = dist.get(bm.rating.value, 0) + 1
        return dist

    def _compute_provenance(self, result: TenantEngagementResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
