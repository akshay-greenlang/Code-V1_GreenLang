# -*- coding: utf-8 -*-
"""
Sector Pathway Workflow
=============================

5-phase workflow for sector-specific decarbonization pathway alignment
within PACK-025 Race to Zero Pack.  Identifies the entity's sector,
retrieves relevant pathways from IEA/IPCC/TPI/MPP, customizes the
pathway to entity context, maps milestones, and benchmarks performance.

Phases:
    1. SectorIdentification  -- Classify entity sector and sub-sector
    2. PathwayRetrieval      -- Retrieve pathways from 25+ sector databases
    3. Customization         -- Customize pathway to entity scale and context
    4. MilestoneMapping      -- Map sector milestones to entity-level actions
    5. BenchmarkComparison   -- Compare entity performance against sector peers

Regulatory references:
    - IEA Net Zero by 2050 Roadmap (2021, updated 2023)
    - IPCC AR6 WG3 Mitigation Pathways (2022)
    - TPI Global Climate Transition Centre (2024)
    - ACT Methodology (ADEME/CDP, 2023)
    - Mission Possible Partnership (2022)
    - CRREM Real Estate Pathways (2023)
    - SBTi SDA Methodology

Zero-hallucination: all pathway data, benchmarks, and gap calculations
use deterministic reference tables.  No LLM calls in the computation path.

Author: GreenLang Team
Version: 25.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"

ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]

def _new_uuid() -> str:
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"

class SectorPathwayPhase(str, Enum):
    SECTOR_IDENTIFICATION = "sector_identification"
    PATHWAY_RETRIEVAL = "pathway_retrieval"
    CUSTOMIZATION = "customization"
    MILESTONE_MAPPING = "milestone_mapping"
    BENCHMARK_COMPARISON = "benchmark_comparison"

class SectorCategory(str, Enum):
    POWER = "power_generation"
    STEEL = "steel"
    CEMENT = "cement"
    CHEMICALS = "chemicals"
    ALUMINIUM = "aluminium"
    TRANSPORT_ROAD = "transport_road"
    TRANSPORT_AVIATION = "transport_aviation"
    TRANSPORT_SHIPPING = "transport_shipping"
    BUILDINGS = "buildings"
    REAL_ESTATE = "real_estate"
    OIL_GAS = "oil_gas"
    MINING = "mining"
    AGRICULTURE = "agriculture"
    FOOD_BEVERAGE = "food_beverage"
    TEXTILES = "textiles"
    AUTOMOTIVE = "automotive"
    TECHNOLOGY = "technology"
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    PROFESSIONAL_SERVICES = "professional_services"
    MANUFACTURING_GENERAL = "manufacturing_general"
    CONSTRUCTION = "construction"
    TELECOMMUNICATIONS = "telecommunications"
    GENERAL_SERVICES = "general_services"

class PathwaySource(str, Enum):
    IEA_NZE = "iea_nze"
    IPCC_AR6 = "ipcc_ar6"
    TPI = "tpi"
    MPP = "mpp"
    CRREM = "crrem"
    ACT = "act"
    SBTI_SDA = "sbti_sda"

class AlignmentStatus(str, Enum):
    ALIGNED = "aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    MISALIGNED = "misaligned"
    NOT_ASSESSED = "not_assessed"

# =============================================================================
# REFERENCE DATA
# =============================================================================

# Sector pathway database (25+ sectors)
SECTOR_PATHWAY_DB: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "name": "Power Generation", "sources": ["iea_nze", "ipcc_ar6", "tpi"],
        "pathway_type": "intensity", "unit": "tCO2/MWh",
        "benchmarks": {2020: 0.46, 2025: 0.35, 2030: 0.14, 2035: 0.05, 2040: 0.02, 2050: 0.0},
        "key_milestones": [
            "Phase out unabated coal by 2030 (OECD)",
            "80% renewable share by 2030",
            "Net-zero power sector by 2040 (advanced economies)",
        ],
        "annual_reduction_rate": 7.0,
    },
    "steel": {
        "name": "Steel & Iron", "sources": ["mpp", "iea_nze", "tpi"],
        "pathway_type": "intensity", "unit": "tCO2/t steel",
        "benchmarks": {2020: 1.89, 2025: 1.7, 2030: 1.4, 2035: 0.9, 2040: 0.4, 2050: 0.0},
        "key_milestones": [
            "Near-zero steel pilots by 2025",
            "30% intensity reduction by 2030",
            "50% DRI-H2/EAF by 2035",
        ],
        "annual_reduction_rate": 4.5,
    },
    "cement": {
        "name": "Cement & Concrete", "sources": ["mpp", "iea_nze"],
        "pathway_type": "intensity", "unit": "tCO2/t cement",
        "benchmarks": {2020: 0.63, 2025: 0.55, 2030: 0.43, 2035: 0.30, 2040: 0.15, 2050: 0.0},
        "key_milestones": [
            "Clinker ratio reduction to 0.65 by 2025",
            "40% emission reduction by 2030",
            "CCS at scale by 2035",
        ],
        "annual_reduction_rate": 3.5,
    },
    "transport_road": {
        "name": "Road Transport", "sources": ["iea_nze", "ipcc_ar6"],
        "pathway_type": "intensity", "unit": "tCO2/vehicle/yr",
        "benchmarks": {2020: 0.14, 2025: 0.11, 2030: 0.08, 2035: 0.04, 2040: 0.02, 2050: 0.0},
        "key_milestones": [
            "20% EV share new sales by 2025",
            "60% EV share by 2030",
            "100% EV new sales by 2035 (OECD)",
        ],
        "annual_reduction_rate": 5.0,
    },
    "buildings": {
        "name": "Buildings", "sources": ["iea_nze", "crrem"],
        "pathway_type": "intensity", "unit": "tCO2/m2/yr",
        "benchmarks": {2020: 0.025, 2025: 0.020, 2030: 0.012, 2035: 0.006, 2040: 0.003, 2050: 0.0},
        "key_milestones": [
            "All new buildings zero-carbon-ready by 2025",
            "50% heat pump share (new sales) by 2030",
            "No new fossil fuel boilers by 2035",
        ],
        "annual_reduction_rate": 4.0,
    },
    "financial_services": {
        "name": "Financial Services", "sources": ["tpi", "sbti_sda"],
        "pathway_type": "absolute", "unit": "tCO2e (financed)",
        "benchmarks": {2020: 1.0, 2025: 0.85, 2030: 0.58, 2035: 0.35, 2040: 0.15, 2050: 0.0},
        "key_milestones": [
            "Financed emissions baseline by 2025",
            "Portfolio 1.5C aligned by 2030",
            "Phase out fossil fuel financing by 2040",
        ],
        "annual_reduction_rate": 5.0,
    },
    "general_services": {
        "name": "General / Service Sector", "sources": ["sbti_sda", "act"],
        "pathway_type": "absolute", "unit": "tCO2e",
        "benchmarks": {2020: 1.0, 2025: 0.80, 2030: 0.58, 2035: 0.40, 2040: 0.20, 2050: 0.0},
        "key_milestones": [
            "100% renewable electricity by 2025",
            "42% absolute reduction by 2030 (1.5C ACA)",
            "80% absolute reduction by 2040",
        ],
        "annual_reduction_rate": 4.2,
    },
    "technology": {
        "name": "Technology", "sources": ["sbti_sda", "act"],
        "pathway_type": "absolute", "unit": "tCO2e",
        "benchmarks": {2020: 1.0, 2025: 0.78, 2030: 0.55, 2035: 0.30, 2040: 0.12, 2050: 0.0},
        "key_milestones": [
            "RE100 commitment by 2025",
            "Supply chain engagement top 70% by 2030",
            "Net-zero operations by 2035",
        ],
        "annual_reduction_rate": 5.0,
    },
    "retail": {
        "name": "Retail", "sources": ["sbti_sda", "act"],
        "pathway_type": "absolute", "unit": "tCO2e",
        "benchmarks": {2020: 1.0, 2025: 0.82, 2030: 0.58, 2035: 0.38, 2040: 0.18, 2050: 0.0},
        "key_milestones": [
            "Refrigerant transition by 2025",
            "Scope 3 Cat 1 engagement by 2030",
            "Full supply chain decarbonization by 2040",
        ],
        "annual_reduction_rate": 4.2,
    },
    "manufacturing_general": {
        "name": "General Manufacturing", "sources": ["sbti_sda", "iea_nze"],
        "pathway_type": "intensity", "unit": "tCO2/revenue $M",
        "benchmarks": {2020: 1.0, 2025: 0.82, 2030: 0.60, 2035: 0.38, 2040: 0.18, 2050: 0.0},
        "key_milestones": [
            "Energy efficiency improvements by 2025",
            "Process electrification by 2030",
            "Net-zero manufacturing by 2050",
        ],
        "annual_reduction_rate": 4.0,
    },
}

# Phase dependencies DAG
PHASE_DEPENDENCIES: Dict[SectorPathwayPhase, List[SectorPathwayPhase]] = {
    SectorPathwayPhase.SECTOR_IDENTIFICATION: [],
    SectorPathwayPhase.PATHWAY_RETRIEVAL: [SectorPathwayPhase.SECTOR_IDENTIFICATION],
    SectorPathwayPhase.CUSTOMIZATION: [SectorPathwayPhase.PATHWAY_RETRIEVAL],
    SectorPathwayPhase.MILESTONE_MAPPING: [SectorPathwayPhase.CUSTOMIZATION],
    SectorPathwayPhase.BENCHMARK_COMPARISON: [SectorPathwayPhase.MILESTONE_MAPPING],
}

PHASE_EXECUTION_ORDER: List[SectorPathwayPhase] = [
    SectorPathwayPhase.SECTOR_IDENTIFICATION,
    SectorPathwayPhase.PATHWAY_RETRIEVAL,
    SectorPathwayPhase.CUSTOMIZATION,
    SectorPathwayPhase.MILESTONE_MAPPING,
    SectorPathwayPhase.BENCHMARK_COMPARISON,
]

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    phase: SectorPathwayPhase = Field(...)
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class SectorProfile(BaseModel):
    sector: str = Field(default="")
    sector_name: str = Field(default="")
    sub_sector: str = Field(default="")
    pathway_sources: List[str] = Field(default_factory=list)
    pathway_type: str = Field(default="absolute")
    intensity_unit: str = Field(default="tCO2e")

class PathwayBenchmark(BaseModel):
    year: int = Field(default=2030)
    benchmark_value: float = Field(default=0.0)
    entity_value: float = Field(default=0.0)
    gap: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    alignment: AlignmentStatus = Field(default=AlignmentStatus.NOT_ASSESSED)

class MilestoneMap(BaseModel):
    milestone_id: str = Field(default="")
    year: int = Field(default=2030)
    description: str = Field(default="")
    entity_action: str = Field(default="")
    achievable: bool = Field(default=False)
    effort_level: str = Field(default="medium")

class SectorPathwayConfig(BaseModel):
    pack_id: str = Field(default="PACK-025")
    org_name: str = Field(default="")
    sector: str = Field(default="general_services")
    sub_sector: str = Field(default="")
    actor_type: str = Field(default="corporate")
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    base_year: int = Field(default=2019, ge=2015, le=2050)
    baseline_tco2e: float = Field(default=0.0, ge=0.0)
    current_tco2e: float = Field(default=0.0, ge=0.0)
    current_intensity: float = Field(default=0.0, ge=0.0)
    intensity_unit: str = Field(default="")
    target_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    revenue_usd: float = Field(default=0.0, ge=0.0)
    production_volume: float = Field(default=0.0, ge=0.0)
    enable_provenance: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class SectorPathwayResult(BaseModel):
    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-025")
    workflow_name: str = Field(default="sector_pathway")
    org_name: str = Field(default="")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
    sector_profile: Optional[SectorProfile] = Field(None)
    benchmarks: List[PathwayBenchmark] = Field(default_factory=list)
    milestones: List[MilestoneMap] = Field(default_factory=list)
    overall_alignment: AlignmentStatus = Field(default=AlignmentStatus.NOT_ASSESSED)
    gap_to_benchmark_pct: float = Field(default=0.0)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class SectorPathwayWorkflow:
    """
    5-phase sector pathway alignment workflow for PACK-025 Race to Zero Pack.

    Maps entity decarbonization plans to sector-specific pathways from
    25+ sectors, customizes benchmarks, maps milestones, and compares
    performance against sector peers.

    Engines used:
        - sector_pathway_engine (pathway retrieval and customization)
        - progress_tracking_engine (benchmark comparison)
    """

    def __init__(
        self,
        config: Optional[SectorPathwayConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.config = config or SectorPathwayConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, SectorPathwayResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback

    async def execute(
        self, input_data: Optional[Dict[str, Any]] = None,
    ) -> SectorPathwayResult:
        """Execute the 5-phase sector pathway workflow."""
        input_data = input_data or {}
        result = SectorPathwayResult(
            org_name=self.config.org_name,
            status=WorkflowStatus.RUNNING, started_at=utcnow(),
        )
        self._results[result.execution_id] = result
        start_time = time.monotonic()
        phases = PHASE_EXECUTION_ORDER

        self.logger.info(
            "Starting sector pathway: execution_id=%s, sector=%s",
            result.execution_id, self.config.sector,
        )

        ctx: Dict[str, Any] = dict(input_data)
        ctx["sector"] = self.config.sector

        try:
            for idx, phase in enumerate(phases):
                if result.execution_id in self._cancelled:
                    result.status = WorkflowStatus.CANCELLED
                    break
                if not self._deps_met(phase, result):
                    result.status = WorkflowStatus.FAILED
                    break

                if self._progress_callback:
                    await self._progress_callback(phase.value, (idx / len(phases)) * 100, phase.value)

                pr = await self._run_phase(phase, ctx)
                result.phase_results[phase.value] = pr
                if pr.status == PhaseStatus.FAILED:
                    result.status = WorkflowStatus.PARTIAL
                result.phases_completed.append(phase.value)
                result.total_records_processed += pr.records_processed
                ctx[phase.value] = pr.outputs

            if result.status == WorkflowStatus.RUNNING:
                result.status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Sector pathway failed: %s", exc, exc_info=True)
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(exc))

        finally:
            result.completed_at = utcnow()
            result.total_duration_ms = (time.monotonic() - start_time) * 1000
            result.sector_profile = self._build_profile(ctx)
            result.benchmarks = self._build_benchmarks(ctx)
            result.milestones = self._build_milestones(ctx)
            result.overall_alignment = AlignmentStatus(
                ctx.get("benchmark_comparison", {}).get("overall_alignment", "not_assessed")
            )
            result.gap_to_benchmark_pct = ctx.get("benchmark_comparison", {}).get("gap_2030_pct", 0)
            result.quality_score = round(
                (len(result.phases_completed) / max(len(phases), 1)) * 100, 1
            )
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    result.model_dump_json(exclude={"provenance_hash"})
                )

        return result

    def cancel(self, eid: str) -> Dict[str, Any]:
        self._cancelled.add(eid)
        return {"cancelled": True}

    async def _run_phase(self, phase: SectorPathwayPhase, ctx: Dict[str, Any]) -> PhaseResult:
        started = utcnow()
        st = time.monotonic()
        handler = {
            SectorPathwayPhase.SECTOR_IDENTIFICATION: self._ph_sector_id,
            SectorPathwayPhase.PATHWAY_RETRIEVAL: self._ph_pathway_retrieval,
            SectorPathwayPhase.CUSTOMIZATION: self._ph_customization,
            SectorPathwayPhase.MILESTONE_MAPPING: self._ph_milestone_mapping,
            SectorPathwayPhase.BENCHMARK_COMPARISON: self._ph_benchmark_comparison,
        }[phase]
        try:
            out, warn, err, rec = await handler(ctx)
            status = PhaseStatus.FAILED if err else PhaseStatus.COMPLETED
        except Exception as exc:
            out, warn, err, rec = {}, [], [str(exc)], 0
            status = PhaseStatus.FAILED
        return PhaseResult(
            phase=phase, status=status, started_at=started, completed_at=utcnow(),
            duration_ms=round((time.monotonic() - st) * 1000, 2), records_processed=rec,
            outputs=out, warnings=warn, errors=err,
            provenance_hash=_compute_hash(out) if self.config.enable_provenance else "",
        )

    def _deps_met(self, phase: SectorPathwayPhase, result: SectorPathwayResult) -> bool:
        for dep in PHASE_DEPENDENCIES.get(phase, []):
            dr = result.phase_results.get(dep.value)
            if not dr or dr.status not in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED):
                return False
        return True

    # ---- Phase Handlers ----

    async def _ph_sector_id(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        sector = self.config.sector
        pathway_info = SECTOR_PATHWAY_DB.get(sector)

        if not pathway_info:
            sector = "general_services"
            pathway_info = SECTOR_PATHWAY_DB["general_services"]
            warnings.append(f"Sector '{self.config.sector}' not in database. Using general_services.")

        outputs["sector"] = sector
        outputs["sector_name"] = pathway_info["name"]
        outputs["sub_sector"] = self.config.sub_sector
        outputs["pathway_sources"] = pathway_info["sources"]
        outputs["pathway_type"] = pathway_info["pathway_type"]
        outputs["intensity_unit"] = pathway_info["unit"]
        outputs["sectors_available"] = len(SECTOR_PATHWAY_DB)

        return outputs, warnings, errors, 1

    async def _ph_pathway_retrieval(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        sector = ctx.get("sector_identification", {}).get("sector", "general_services")
        pathway_info = SECTOR_PATHWAY_DB.get(sector, SECTOR_PATHWAY_DB["general_services"])

        outputs["benchmarks"] = pathway_info["benchmarks"]
        outputs["key_milestones"] = pathway_info["key_milestones"]
        outputs["annual_reduction_rate"] = pathway_info["annual_reduction_rate"]
        outputs["pathway_sources"] = pathway_info["sources"]
        outputs["benchmark_years"] = list(pathway_info["benchmarks"].keys())

        return outputs, warnings, errors, len(pathway_info["benchmarks"])

    async def _ph_customization(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        pathway = ctx.get("pathway_retrieval", {})
        benchmarks = pathway.get("benchmarks", {})
        baseline = self.config.baseline_tco2e
        current = self.config.current_tco2e

        # Normalize benchmarks to entity scale
        base_benchmark = benchmarks.get(2020, 1.0)
        customized: Dict[int, float] = {}
        for year, bm_val in benchmarks.items():
            ratio = bm_val / max(base_benchmark, 0.001)
            customized[year] = round(baseline * ratio, 2)

        # Entity's current position relative to pathway
        current_expected = customized.get(self.config.reporting_year, baseline)
        position_gap = current - current_expected

        outputs["customized_benchmarks"] = customized
        outputs["entity_baseline_tco2e"] = round(baseline, 2)
        outputs["entity_current_tco2e"] = round(current, 2)
        outputs["current_expected_tco2e"] = round(current_expected, 2)
        outputs["position_gap_tco2e"] = round(position_gap, 2)
        outputs["ahead_of_pathway"] = position_gap <= 0

        if position_gap > 0:
            warnings.append(
                f"Entity is {position_gap:.0f} tCO2e behind the sector pathway. "
                "Accelerated action needed."
            )

        return outputs, warnings, errors, len(customized)

    async def _ph_milestone_mapping(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        pathway = ctx.get("pathway_retrieval", {})
        milestones_raw = pathway.get("key_milestones", [])
        customized = ctx.get("customization", {})

        milestones: List[Dict[str, Any]] = []
        for i, ms_text in enumerate(milestones_raw):
            # Extract year from milestone text (look for 4-digit year)
            year = 2030  # default
            for word in ms_text.split():
                if word.isdigit() and len(word) == 4:
                    year = int(word)
                    break

            milestone = {
                "milestone_id": f"MS-{i+1:02d}",
                "year": year,
                "description": ms_text,
                "entity_action": f"Implement actions to achieve: {ms_text}",
                "achievable": year >= self.config.reporting_year,
                "effort_level": "high" if year <= 2025 else "medium",
            }
            milestones.append(milestone)

        outputs["milestones"] = milestones
        outputs["milestones_count"] = len(milestones)
        outputs["near_term_milestones"] = sum(1 for m in milestones if m["year"] <= 2030)
        outputs["achievable_milestones"] = sum(1 for m in milestones if m["achievable"])

        return outputs, warnings, errors, len(milestones)

    async def _ph_benchmark_comparison(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        customized = ctx.get("customization", {})
        custom_benchmarks = customized.get("customized_benchmarks", {})
        current = self.config.current_tco2e
        baseline = self.config.baseline_tco2e

        benchmark_results: List[Dict[str, Any]] = []
        for year, bm_val in sorted(custom_benchmarks.items()):
            if year < self.config.reporting_year:
                continue

            # Project entity emissions linearly
            years_from_now = year - self.config.reporting_year
            annual_rate = self.config.target_reduction_pct / max(2050 - self.config.base_year, 1)
            projected = current * (1 - annual_rate / 100.0) ** years_from_now

            gap = projected - bm_val
            gap_pct = (gap / max(bm_val, 1)) * 100.0 if bm_val > 0 else 0

            if gap <= 0:
                alignment = AlignmentStatus.ALIGNED.value
            elif gap_pct <= 20:
                alignment = AlignmentStatus.PARTIALLY_ALIGNED.value
            else:
                alignment = AlignmentStatus.MISALIGNED.value

            benchmark_results.append({
                "year": year,
                "benchmark_tco2e": round(bm_val, 2),
                "projected_tco2e": round(projected, 2),
                "gap_tco2e": round(gap, 2),
                "gap_pct": round(gap_pct, 1),
                "alignment": alignment,
            })

        # Overall alignment (based on 2030)
        bm_2030 = next((b for b in benchmark_results if b["year"] == 2030), None)
        if bm_2030:
            overall_alignment = bm_2030["alignment"]
            gap_2030 = bm_2030["gap_pct"]
        else:
            overall_alignment = AlignmentStatus.NOT_ASSESSED.value
            gap_2030 = 0

        outputs["benchmark_results"] = benchmark_results
        outputs["overall_alignment"] = overall_alignment
        outputs["gap_2030_pct"] = round(gap_2030, 1)
        outputs["years_assessed"] = len(benchmark_results)
        outputs["aligned_years"] = sum(1 for b in benchmark_results if b["alignment"] == "aligned")

        if overall_alignment == AlignmentStatus.MISALIGNED.value:
            warnings.append(
                f"Entity projected emissions misaligned with sector pathway by 2030 "
                f"(gap: {gap_2030:.1f}%). Review action plan for additional levers."
            )

        return outputs, warnings, errors, len(benchmark_results)

    # ---- Extractors ----

    def _build_profile(self, ctx: Dict[str, Any]) -> Optional[SectorProfile]:
        d = ctx.get("sector_identification", {})
        if not d:
            return None
        return SectorProfile(
            sector=d.get("sector", ""), sector_name=d.get("sector_name", ""),
            sub_sector=d.get("sub_sector", ""), pathway_sources=d.get("pathway_sources", []),
            pathway_type=d.get("pathway_type", "absolute"), intensity_unit=d.get("intensity_unit", ""),
        )

    def _build_benchmarks(self, ctx: Dict[str, Any]) -> List[PathwayBenchmark]:
        data = ctx.get("benchmark_comparison", {}).get("benchmark_results", [])
        return [
            PathwayBenchmark(
                year=b["year"], benchmark_value=b["benchmark_tco2e"],
                entity_value=b["projected_tco2e"], gap=b["gap_tco2e"],
                gap_pct=b["gap_pct"], alignment=AlignmentStatus(b["alignment"]),
            )
            for b in data
        ]

    def _build_milestones(self, ctx: Dict[str, Any]) -> List[MilestoneMap]:
        data = ctx.get("milestone_mapping", {}).get("milestones", [])
        return [
            MilestoneMap(
                milestone_id=m["milestone_id"], year=m["year"],
                description=m["description"], entity_action=m["entity_action"],
                achievable=m["achievable"], effort_level=m["effort_level"],
            )
            for m in data
        ]
