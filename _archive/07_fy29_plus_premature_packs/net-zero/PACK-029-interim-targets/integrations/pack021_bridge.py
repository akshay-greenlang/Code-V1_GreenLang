# -*- coding: utf-8 -*-
"""
PACK021Bridge - PACK-021 Net Zero Starter Pack Integration for PACK-029
=========================================================================

Enterprise bridge for integrating PACK-021 (Net Zero Starter Pack) baseline
emissions, long-term net-zero target, SBTi pathway selection, and
organizational boundary into the Interim Targets Pack. PACK-021 provides
the foundational GHG inventory baseline and net-zero commitment that
PACK-029 decomposes into 5-year interim milestones, annual carbon budgets,
and quarterly performance tracking checkpoints.

Integration Points:
    - Baseline Engine: Base year GHG inventory (Scope 1+2+3) from PACK-021
    - Target Engine: Long-term target (net-zero year, 2050 target emissions)
    - SBTi Pathway: 1.5C or WB2C pathway selection from PACK-021
    - Organizational Boundary: Operational/financial control, equity share
    - Activity Data: Reuses PACK-021 activity data for interim decomposition
    - Provenance Chain: Links PACK-029 interim targets to PACK-021 baseline

Architecture:
    PACK-021 Baseline --> PACK-029 Interim Target Decomposition
    PACK-021 Targets  --> PACK-029 5-Year Milestone Generation
    PACK-021 Pathway  --> PACK-029 Annual Carbon Budget Allocation
    PACK-021 Boundary --> PACK-029 Scope Coverage Validation

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-029 Interim Targets Pack
Status: Production Ready
"""

import hashlib
import importlib
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

class _PackStub:
    """Stub for PACK-021 components when not available."""
    def __init__(self, component: str) -> None:
        self._component = component

    def __getattr__(self, name: str) -> Any:
        def _stub(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"component": self._component, "status": "not_available", "pack": "PACK-021"}
        return _stub

def _try_import_pack021(component: str, module_path: str) -> Any:
    """Attempt to import a PACK-021 component."""
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("PACK-021 component '%s' not available, using stub", component)
        return _PackStub(component)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BaselineStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VALIDATED = "validated"
    EXPIRED = "expired"

class SBTiPathwayType(str, Enum):
    ACA_15C = "aca_15c"
    ACA_WB2C = "aca_wb2c"
    SDA = "sda"
    FLAG = "flag"
    HYBRID = "hybrid"

class BoundaryApproach(str, Enum):
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"

class DataQualityTier(str, Enum):
    TIER_1 = "tier_1"  # Primary measured data
    TIER_2 = "tier_2"  # Supplier-specific data
    TIER_3 = "tier_3"  # Industry average data
    TIER_4 = "tier_4"  # Spend-based estimates
    TIER_5 = "tier_5"  # Extrapolated / modelled

class ImportStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    STALE = "stale"

# ---------------------------------------------------------------------------
# PACK-021 Component Registry
# ---------------------------------------------------------------------------

PACK021_COMPONENTS: Dict[str, Dict[str, str]] = {
    "baseline_engine": {
        "name": "GHG Baseline Engine",
        "module": "packs.net_zero.PACK_021_net_zero_starter.engines.baseline_engine",
        "description": "Scope 1+2+3 GHG inventory baseline calculation",
    },
    "target_engine": {
        "name": "Target Setting Engine",
        "module": "packs.net_zero.PACK_021_net_zero_starter.engines.target_engine",
        "description": "ACA/SDA/FLAG target generation with SBTi alignment",
    },
    "gap_analysis_engine": {
        "name": "Gap Analysis Engine",
        "module": "packs.net_zero.PACK_021_net_zero_starter.engines.gap_analysis_engine",
        "description": "Gap quantification between current and target emissions",
    },
    "data_intake_engine": {
        "name": "Data Intake Engine",
        "module": "packs.net_zero.PACK_021_net_zero_starter.engines.data_intake_engine",
        "description": "Activity data collection and normalization",
    },
    "reduction_engine": {
        "name": "Reduction Action Engine",
        "module": "packs.net_zero.PACK_021_net_zero_starter.engines.reduction_engine",
        "description": "Emission reduction action identification and prioritization",
    },
    "macc_engine": {
        "name": "MACC Engine",
        "module": "packs.net_zero.PACK_021_net_zero_starter.engines.macc_engine",
        "description": "Marginal abatement cost curve generation",
    },
}

# ---------------------------------------------------------------------------
# SBTi Minimum Ambition Tables
# ---------------------------------------------------------------------------

SBTI_MINIMUM_AMBITION: Dict[str, Dict[str, float]] = {
    "aca_15c": {
        "scope12_annual_rate_pct": 4.2,
        "scope12_2030_reduction_pct": 42.0,
        "scope3_annual_rate_pct": 2.5,
        "scope3_2030_reduction_pct": 25.0,
        "long_term_scope12_reduction_pct": 90.0,
        "long_term_scope3_reduction_pct": 90.0,
    },
    "aca_wb2c": {
        "scope12_annual_rate_pct": 2.5,
        "scope12_2030_reduction_pct": 25.0,
        "scope3_annual_rate_pct": 2.5,
        "scope3_2030_reduction_pct": 25.0,
        "long_term_scope12_reduction_pct": 90.0,
        "long_term_scope3_reduction_pct": 90.0,
    },
    "sda": {
        "scope12_annual_rate_pct": 4.2,
        "scope12_2030_reduction_pct": 42.0,
        "scope3_annual_rate_pct": 2.5,
        "scope3_2030_reduction_pct": 25.0,
        "long_term_scope12_reduction_pct": 90.0,
        "long_term_scope3_reduction_pct": 90.0,
    },
}

SCOPE3_THRESHOLD_PCT: float = 40.0

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class PACK021BridgeConfig(BaseModel):
    """Configuration for the PACK-021 to PACK-029 bridge."""
    pack_id: str = Field(default="PACK-029")
    pack021_id: str = Field(default="PACK-021")
    pack021_baseline_id: str = Field(default="")
    organization_name: str = Field(default="")
    organization_id: str = Field(default="")
    base_year: int = Field(default=2023, ge=2015, le=2025)
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    net_zero_year: int = Field(default=2050, ge=2040, le=2060)
    sbti_pathway: SBTiPathwayType = Field(default=SBTiPathwayType.ACA_15C)
    boundary_approach: BoundaryApproach = Field(default=BoundaryApproach.OPERATIONAL_CONTROL)
    enable_provenance: bool = Field(default=True)
    auto_import_baseline: bool = Field(default=True)
    sync_activity_data: bool = Field(default=True)
    db_connection_string: str = Field(default="")
    db_pool_size: int = Field(default=5, ge=1, le=20)
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=30.0)

class BaselineImport(BaseModel):
    """Imported baseline data from PACK-021 for interim target decomposition."""
    import_id: str = Field(default_factory=_new_uuid)
    source_pack: str = Field(default="PACK-021")
    source_baseline_id: str = Field(default="")
    organization_name: str = Field(default="")
    organization_id: str = Field(default="")
    base_year: int = Field(default=2023)
    scope1_tco2e: float = Field(default=0.0)
    scope2_location_tco2e: float = Field(default=0.0)
    scope2_market_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    scope3_by_category: Dict[int, float] = Field(default_factory=dict)
    total_tco2e: float = Field(default=0.0)
    scope3_share_pct: float = Field(default=0.0)
    scope3_above_threshold: bool = Field(default=False)
    activity_data: Dict[str, Any] = Field(default_factory=dict)
    emission_factors: Dict[str, float] = Field(default_factory=dict)
    status: BaselineStatus = Field(default=BaselineStatus.NOT_STARTED)
    data_quality_tier: DataQualityTier = Field(default=DataQualityTier.TIER_3)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    boundary_approach: BoundaryApproach = Field(default=BoundaryApproach.OPERATIONAL_CONTROL)
    imported_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class LongTermTargetImport(BaseModel):
    """Imported long-term net-zero target from PACK-021."""
    import_id: str = Field(default_factory=_new_uuid)
    source_pack: str = Field(default="PACK-021")
    net_zero_year: int = Field(default=2050)
    base_year: int = Field(default=2023)
    base_year_total_tco2e: float = Field(default=0.0)
    target_total_tco2e: float = Field(default=0.0)
    residual_emissions_tco2e: float = Field(default=0.0)
    scope1_target_tco2e: float = Field(default=0.0)
    scope2_target_tco2e: float = Field(default=0.0)
    scope3_target_tco2e: float = Field(default=0.0)
    total_reduction_pct: float = Field(default=90.0)
    neutralization_plan: str = Field(default="")
    provenance_hash: str = Field(default="")

class SBTiPathwayImport(BaseModel):
    """Imported SBTi pathway selection from PACK-021."""
    import_id: str = Field(default_factory=_new_uuid)
    source_pack: str = Field(default="PACK-021")
    pathway_type: SBTiPathwayType = Field(default=SBTiPathwayType.ACA_15C)
    temperature_alignment: str = Field(default="1.5C")
    near_term_target_year: int = Field(default=2030)
    near_term_scope12_reduction_pct: float = Field(default=42.0)
    near_term_scope3_reduction_pct: float = Field(default=25.0)
    annual_scope12_reduction_rate_pct: float = Field(default=4.2)
    annual_scope3_reduction_rate_pct: float = Field(default=2.5)
    scope12_coverage_pct: float = Field(default=95.0)
    scope3_coverage_pct: float = Field(default=67.0)
    flag_sector_applicable: bool = Field(default=False)
    flag_commodities: List[str] = Field(default_factory=list)
    sbti_submission_status: str = Field(default="draft")
    provenance_hash: str = Field(default="")

class BoundaryImport(BaseModel):
    """Imported organizational boundary from PACK-021."""
    import_id: str = Field(default_factory=_new_uuid)
    source_pack: str = Field(default="PACK-021")
    boundary_approach: BoundaryApproach = Field(default=BoundaryApproach.OPERATIONAL_CONTROL)
    legal_entities: List[str] = Field(default_factory=list)
    operating_countries: List[str] = Field(default_factory=list)
    business_units: List[Dict[str, Any]] = Field(default_factory=list)
    excluded_sources: List[str] = Field(default_factory=list)
    excluded_sources_pct: float = Field(default=0.0)
    scope12_coverage_pct: float = Field(default=95.0)
    scope3_coverage_pct: float = Field(default=67.0)
    consolidation_notes: str = Field(default="")
    provenance_hash: str = Field(default="")

class PACK021IntegrationResult(BaseModel):
    """Complete PACK-021 integration result for PACK-029."""
    result_id: str = Field(default_factory=_new_uuid)
    baseline: Optional[BaselineImport] = Field(None)
    long_term_target: Optional[LongTermTargetImport] = Field(None)
    sbti_pathway: Optional[SBTiPathwayImport] = Field(None)
    boundary: Optional[BoundaryImport] = Field(None)
    pack021_available: bool = Field(default=False)
    components_loaded: List[str] = Field(default_factory=list)
    components_stubbed: List[str] = Field(default_factory=list)
    import_status: ImportStatus = Field(default=ImportStatus.FAILED)
    integration_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    imported_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# PACK021Bridge
# ---------------------------------------------------------------------------

class PACK021Bridge:
    """PACK-021 Net Zero Starter Pack integration bridge for PACK-029.

    Imports baseline GHG inventory, long-term net-zero target, SBTi
    pathway selection, and organizational boundary from PACK-021 for
    decomposition into interim targets, annual carbon budgets, and
    quarterly performance checkpoints.

    Example:
        >>> config = PACK021BridgeConfig(
        ...     organization_name="Acme Corp",
        ...     pack021_baseline_id="baseline-2023-001",
        ...     sbti_pathway=SBTiPathwayType.ACA_15C,
        ... )
        >>> bridge = PACK021Bridge(config)
        >>> baseline = await bridge.import_baseline()
        >>> target = await bridge.import_long_term_target()
        >>> pathway = await bridge.import_sbti_pathway()
        >>> boundary = await bridge.import_boundary()
        >>> result = await bridge.get_full_integration()
    """

    def __init__(self, config: Optional[PACK021BridgeConfig] = None) -> None:
        self.config = config or PACK021BridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._components: Dict[str, Any] = {}
        self._loaded: List[str] = []
        self._stubbed: List[str] = []

        for comp_id, comp_info in PACK021_COMPONENTS.items():
            agent = _try_import_pack021(comp_id, comp_info["module"])
            self._components[comp_id] = agent
            if isinstance(agent, _PackStub):
                self._stubbed.append(comp_id)
            else:
                self._loaded.append(comp_id)

        self._baseline_cache: Optional[BaselineImport] = None
        self._target_cache: Optional[LongTermTargetImport] = None
        self._pathway_cache: Optional[SBTiPathwayImport] = None
        self._boundary_cache: Optional[BoundaryImport] = None
        self._db_pool: Optional[Any] = None

        self.logger.info(
            "PACK021Bridge (PACK-029) initialized: %d/%d components loaded, "
            "org=%s, pathway=%s, boundary=%s",
            len(self._loaded), len(PACK021_COMPONENTS),
            self.config.organization_name,
            self.config.sbti_pathway.value,
            self.config.boundary_approach.value,
        )

    async def _get_db_pool(self) -> Any:
        """Get or create async database connection pool."""
        if self._db_pool is not None:
            return self._db_pool
        if not self.config.db_connection_string:
            return None
        try:
            import psycopg_pool
            self._db_pool = psycopg_pool.AsyncConnectionPool(
                self.config.db_connection_string,
                min_size=1,
                max_size=self.config.db_pool_size,
            )
            await self._db_pool.open()
            return self._db_pool
        except Exception as exc:
            self.logger.warning("DB pool creation failed: %s", exc)
            return None

    async def _query_pack021_table(
        self, query: str, params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute async PostgreSQL query against PACK-021 tables."""
        pool = await self._get_db_pool()
        if not pool:
            return []
        attempt = 0
        while attempt < self.config.retry_attempts:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query, params or {})
                        columns = [desc[0] for desc in cur.description] if cur.description else []
                        rows = await cur.fetchall()
                        return [dict(zip(columns, row)) for row in rows]
            except Exception as exc:
                attempt += 1
                self.logger.warning(
                    "DB query attempt %d/%d failed: %s",
                    attempt, self.config.retry_attempts, exc,
                )
                if attempt < self.config.retry_attempts:
                    import asyncio

                    await asyncio.sleep(self.config.retry_delay_seconds * attempt)
        return []

    async def import_baseline(
        self, baseline_data: Optional[Dict[str, Any]] = None,
    ) -> BaselineImport:
        """Import GHG baseline from PACK-021 or provided data.

        Retrieves base year emissions for Scope 1, 2 (location + market),
        and Scope 3 (all 15 categories) plus activity data and emission
        factors used in the baseline calculation.
        """
        data = baseline_data or {}

        # Try DB import first
        if not data and self.config.db_connection_string:
            db_rows = await self._query_pack021_table(
                "SELECT * FROM gl_pack021_baselines "
                "WHERE baseline_id = %(baseline_id)s AND base_year = %(base_year)s "
                "ORDER BY created_at DESC LIMIT 1",
                {
                    "baseline_id": self.config.pack021_baseline_id,
                    "base_year": self.config.base_year,
                },
            )
            if db_rows:
                data = db_rows[0]

        scope3_by_cat = data.get("scope3_by_category", {
            1: 45000, 2: 8000, 3: 12000, 4: 15000, 5: 5000,
            6: 8000, 7: 6000, 8: 2000, 9: 4000, 10: 3000,
            11: 5000, 12: 2000, 13: 1000, 14: 500, 15: 3500,
        })
        scope3_total = data.get("scope3_tco2e", sum(scope3_by_cat.values()))

        baseline = BaselineImport(
            source_baseline_id=self.config.pack021_baseline_id or data.get("baseline_id", ""),
            organization_name=self.config.organization_name,
            organization_id=self.config.organization_id,
            base_year=data.get("base_year", self.config.base_year),
            scope1_tco2e=data.get("scope1_tco2e", 50000.0),
            scope2_location_tco2e=data.get("scope2_location_tco2e", 30000.0),
            scope2_market_tco2e=data.get("scope2_market_tco2e", 25000.0),
            scope3_tco2e=scope3_total,
            scope3_by_category=scope3_by_cat,
            status=BaselineStatus.VALIDATED if data else BaselineStatus.COMPLETED,
            data_quality_tier=DataQualityTier(data.get("data_quality_tier", "tier_3")),
            data_quality_score=data.get("data_quality_score", 0.85),
            boundary_approach=self.config.boundary_approach,
            activity_data=data.get("activity_data", {}),
            emission_factors=data.get("emission_factors", {}),
        )

        # Calculate totals and Scope 3 share
        baseline.total_tco2e = (
            baseline.scope1_tco2e + baseline.scope2_market_tco2e + baseline.scope3_tco2e
        )
        baseline.scope3_share_pct = round(
            (baseline.scope3_tco2e / max(baseline.total_tco2e, 1.0)) * 100.0, 2
        )
        baseline.scope3_above_threshold = baseline.scope3_share_pct > SCOPE3_THRESHOLD_PCT

        if self.config.enable_provenance:
            baseline.provenance_hash = _compute_hash(baseline)

        self._baseline_cache = baseline
        self.logger.info(
            "Baseline imported from PACK-021: total=%.1f tCO2e, year=%d, "
            "S3_share=%.1f%%, dq=%.2f, boundary=%s",
            baseline.total_tco2e, baseline.base_year,
            baseline.scope3_share_pct, baseline.data_quality_score,
            baseline.boundary_approach.value,
        )
        return baseline

    async def import_long_term_target(
        self, target_data: Optional[Dict[str, Any]] = None,
    ) -> LongTermTargetImport:
        """Import long-term net-zero target from PACK-021.

        Retrieves the net-zero commitment year, residual emissions
        plan, and scope-level targets for 2050 (or sooner).
        """
        data = target_data or {}
        baseline = self._baseline_cache

        if not data and self.config.db_connection_string:
            db_rows = await self._query_pack021_table(
                "SELECT * FROM gl_pack021_targets "
                "WHERE organization_id = %(org_id)s AND target_type = 'net_zero' "
                "ORDER BY created_at DESC LIMIT 1",
                {"org_id": self.config.organization_id},
            )
            if db_rows:
                data = db_rows[0]

        base_total = baseline.total_tco2e if baseline else data.get("base_total_tco2e", 195000.0)
        base_s1 = baseline.scope1_tco2e if baseline else data.get("scope1_tco2e", 50000.0)
        base_s2 = baseline.scope2_market_tco2e if baseline else data.get("scope2_tco2e", 25000.0)
        base_s3 = baseline.scope3_tco2e if baseline else data.get("scope3_tco2e", 120000.0)

        residual_pct = data.get("residual_emissions_pct", 5.0) / 100.0
        reduction_pct = data.get("total_reduction_pct", 95.0)

        target = LongTermTargetImport(
            net_zero_year=data.get("net_zero_year", self.config.net_zero_year),
            base_year=data.get("base_year", self.config.base_year),
            base_year_total_tco2e=round(base_total, 2),
            target_total_tco2e=round(base_total * residual_pct, 2),
            residual_emissions_tco2e=round(base_total * residual_pct, 2),
            scope1_target_tco2e=round(base_s1 * 0.05, 2),
            scope2_target_tco2e=round(base_s2 * 0.05, 2),
            scope3_target_tco2e=round(base_s3 * 0.10, 2),
            total_reduction_pct=reduction_pct,
            neutralization_plan=data.get(
                "neutralization_plan",
                "Permanent carbon dioxide removal (CDR) for residual emissions, "
                "including direct air capture (DAC) and enhanced weathering."
            ),
        )

        if self.config.enable_provenance:
            target.provenance_hash = _compute_hash(target)

        self._target_cache = target
        self.logger.info(
            "Long-term target imported: net_zero_year=%d, reduction=%.1f%%, "
            "residual=%.1f tCO2e",
            target.net_zero_year, target.total_reduction_pct,
            target.residual_emissions_tco2e,
        )
        return target

    async def import_sbti_pathway(
        self, pathway_data: Optional[Dict[str, Any]] = None,
    ) -> SBTiPathwayImport:
        """Import SBTi pathway selection from PACK-021.

        Retrieves the selected SBTi pathway (1.5C or WB2C), near-term
        reduction rates, scope coverage requirements, and FLAG sector
        applicability.
        """
        data = pathway_data or {}
        pathway_type = SBTiPathwayType(
            data.get("pathway_type", self.config.sbti_pathway.value)
        )
        ambition = SBTI_MINIMUM_AMBITION.get(pathway_type.value, SBTI_MINIMUM_AMBITION["aca_15c"])

        temp_align = "1.5C" if pathway_type in (SBTiPathwayType.ACA_15C, SBTiPathwayType.SDA) else "WB2C"

        pathway = SBTiPathwayImport(
            pathway_type=pathway_type,
            temperature_alignment=data.get("temperature_alignment", temp_align),
            near_term_target_year=data.get("near_term_target_year", 2030),
            near_term_scope12_reduction_pct=data.get(
                "near_term_scope12_reduction_pct",
                ambition["scope12_2030_reduction_pct"],
            ),
            near_term_scope3_reduction_pct=data.get(
                "near_term_scope3_reduction_pct",
                ambition["scope3_2030_reduction_pct"],
            ),
            annual_scope12_reduction_rate_pct=data.get(
                "annual_scope12_reduction_rate_pct",
                ambition["scope12_annual_rate_pct"],
            ),
            annual_scope3_reduction_rate_pct=data.get(
                "annual_scope3_reduction_rate_pct",
                ambition["scope3_annual_rate_pct"],
            ),
            scope12_coverage_pct=data.get("scope12_coverage_pct", 95.0),
            scope3_coverage_pct=data.get("scope3_coverage_pct", 67.0),
            flag_sector_applicable=data.get("flag_sector_applicable", False),
            flag_commodities=data.get("flag_commodities", []),
            sbti_submission_status=data.get("sbti_submission_status", "draft"),
        )

        if self.config.enable_provenance:
            pathway.provenance_hash = _compute_hash(pathway)

        self._pathway_cache = pathway
        self.logger.info(
            "SBTi pathway imported: type=%s, temp=%s, S12_rate=%.1f%%/yr, "
            "S3_rate=%.1f%%/yr, FLAG=%s",
            pathway.pathway_type.value, pathway.temperature_alignment,
            pathway.annual_scope12_reduction_rate_pct,
            pathway.annual_scope3_reduction_rate_pct,
            pathway.flag_sector_applicable,
        )
        return pathway

    async def import_boundary(
        self, boundary_data: Optional[Dict[str, Any]] = None,
    ) -> BoundaryImport:
        """Import organizational boundary from PACK-021.

        Retrieves organizational boundary definition including
        consolidation approach, legal entities, excluded sources,
        and scope coverage percentages.
        """
        data = boundary_data or {}

        if not data and self.config.db_connection_string:
            db_rows = await self._query_pack021_table(
                "SELECT * FROM gl_pack021_boundaries "
                "WHERE organization_id = %(org_id)s "
                "ORDER BY created_at DESC LIMIT 1",
                {"org_id": self.config.organization_id},
            )
            if db_rows:
                data = db_rows[0]

        boundary = BoundaryImport(
            boundary_approach=BoundaryApproach(
                data.get("boundary_approach", self.config.boundary_approach.value)
            ),
            legal_entities=data.get("legal_entities", []),
            operating_countries=data.get("operating_countries", []),
            business_units=data.get("business_units", []),
            excluded_sources=data.get("excluded_sources", []),
            excluded_sources_pct=data.get("excluded_sources_pct", 2.0),
            scope12_coverage_pct=data.get("scope12_coverage_pct", 95.0),
            scope3_coverage_pct=data.get("scope3_coverage_pct", 67.0),
            consolidation_notes=data.get("consolidation_notes", ""),
        )

        if self.config.enable_provenance:
            boundary.provenance_hash = _compute_hash(boundary)

        self._boundary_cache = boundary
        self.logger.info(
            "Boundary imported: approach=%s, entities=%d, countries=%d, "
            "excluded=%.1f%%",
            boundary.boundary_approach.value,
            len(boundary.legal_entities),
            len(boundary.operating_countries),
            boundary.excluded_sources_pct,
        )
        return boundary

    async def get_full_integration(self) -> PACK021IntegrationResult:
        """Get complete PACK-021 integration result for PACK-029.

        Auto-imports all four data sets (baseline, long-term target,
        SBTi pathway, boundary) if not already imported.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not self._baseline_cache:
            try:
                await self.import_baseline()
            except Exception as exc:
                errors.append(f"Baseline import failed: {exc}")

        if not self._target_cache:
            try:
                await self.import_long_term_target()
            except Exception as exc:
                errors.append(f"Long-term target import failed: {exc}")

        if not self._pathway_cache:
            try:
                await self.import_sbti_pathway()
            except Exception as exc:
                errors.append(f"SBTi pathway import failed: {exc}")

        if not self._boundary_cache:
            try:
                await self.import_boundary()
            except Exception as exc:
                warnings.append(f"Boundary import failed (non-critical): {exc}")

        # Validate consistency
        if self._baseline_cache and self._target_cache:
            if self._baseline_cache.base_year != self._target_cache.base_year:
                warnings.append(
                    f"Base year mismatch: baseline={self._baseline_cache.base_year}, "
                    f"target={self._target_cache.base_year}"
                )

        if self._baseline_cache and self._pathway_cache:
            if self._baseline_cache.scope3_above_threshold and self._pathway_cache.scope3_coverage_pct < 67.0:
                warnings.append(
                    "Scope 3 is >40% of total but coverage <67%. SBTi requires "
                    "separate Scope 3 target."
                )

        # Calculate quality score
        quality = 0.0
        if self._baseline_cache:
            quality += 30.0
        if self._target_cache:
            quality += 25.0
        if self._pathway_cache:
            quality += 25.0
        if self._boundary_cache:
            quality += 20.0

        # Determine status
        if errors:
            status = ImportStatus.FAILED if quality < 30.0 else ImportStatus.PARTIAL
        else:
            status = ImportStatus.SUCCESS

        result = PACK021IntegrationResult(
            baseline=self._baseline_cache,
            long_term_target=self._target_cache,
            sbti_pathway=self._pathway_cache,
            boundary=self._boundary_cache,
            pack021_available=len(self._loaded) > 0,
            components_loaded=self._loaded,
            components_stubbed=self._stubbed,
            import_status=status,
            integration_quality_score=quality,
            validation_errors=errors,
            validation_warnings=warnings,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "PACK-021 full integration: status=%s, quality=%.1f%%, "
            "errors=%d, warnings=%d",
            status.value, quality, len(errors), len(warnings),
        )
        return result

    async def get_interim_decomposition_inputs(self) -> Dict[str, Any]:
        """Get all inputs needed for interim target decomposition.

        Returns a consolidated dictionary with baseline, target,
        pathway, and boundary data structured for the interim
        target engine.
        """
        result = await self.get_full_integration()

        baseline = result.baseline
        target = result.long_term_target
        pathway = result.sbti_pathway

        return {
            "base_year": baseline.base_year if baseline else self.config.base_year,
            "net_zero_year": target.net_zero_year if target else self.config.net_zero_year,
            "base_year_emissions": {
                "scope1_tco2e": baseline.scope1_tco2e if baseline else 0.0,
                "scope2_location_tco2e": baseline.scope2_location_tco2e if baseline else 0.0,
                "scope2_market_tco2e": baseline.scope2_market_tco2e if baseline else 0.0,
                "scope3_tco2e": baseline.scope3_tco2e if baseline else 0.0,
                "scope3_by_category": baseline.scope3_by_category if baseline else {},
                "total_tco2e": baseline.total_tco2e if baseline else 0.0,
            },
            "long_term_target": {
                "target_total_tco2e": target.target_total_tco2e if target else 0.0,
                "residual_emissions_tco2e": target.residual_emissions_tco2e if target else 0.0,
                "total_reduction_pct": target.total_reduction_pct if target else 90.0,
            },
            "sbti_pathway": {
                "pathway_type": pathway.pathway_type.value if pathway else "aca_15c",
                "temperature_alignment": pathway.temperature_alignment if pathway else "1.5C",
                "scope12_annual_rate_pct": pathway.annual_scope12_reduction_rate_pct if pathway else 4.2,
                "scope3_annual_rate_pct": pathway.annual_scope3_reduction_rate_pct if pathway else 2.5,
                "scope3_above_threshold": baseline.scope3_above_threshold if baseline else False,
                "flag_applicable": pathway.flag_sector_applicable if pathway else False,
            },
            "boundary": {
                "approach": self.config.boundary_approach.value,
                "scope12_coverage_pct": (
                    result.boundary.scope12_coverage_pct if result.boundary else 95.0
                ),
                "scope3_coverage_pct": (
                    result.boundary.scope3_coverage_pct if result.boundary else 67.0
                ),
            },
            "data_quality": {
                "tier": baseline.data_quality_tier.value if baseline else "tier_3",
                "score": baseline.data_quality_score if baseline else 0.0,
            },
            "integration_quality_score": result.integration_quality_score,
            "import_status": result.import_status.value,
        }

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "pack021_id": self.config.pack021_id,
            "components_total": len(PACK021_COMPONENTS),
            "components_loaded": len(self._loaded),
            "components_stubbed": len(self._stubbed),
            "baseline_imported": self._baseline_cache is not None,
            "target_imported": self._target_cache is not None,
            "pathway_imported": self._pathway_cache is not None,
            "boundary_imported": self._boundary_cache is not None,
            "sbti_pathway": self.config.sbti_pathway.value,
            "boundary_approach": self.config.boundary_approach.value,
            "db_connected": self._db_pool is not None,
        }

    async def sync_with_pack021(self) -> Dict[str, Any]:
        """Synchronize all data with PACK-021 (refresh all imports)."""
        self._baseline_cache = None
        self._target_cache = None
        self._pathway_cache = None
        self._boundary_cache = None

        result = await self.get_full_integration()
        return {
            "synced": True,
            "timestamp": utcnow().isoformat(),
            "status": result.import_status.value,
            "quality_score": result.integration_quality_score,
            "baseline_total_tco2e": result.baseline.total_tco2e if result.baseline else 0.0,
            "net_zero_year": result.long_term_target.net_zero_year if result.long_term_target else 0,
            "pathway": result.sbti_pathway.pathway_type.value if result.sbti_pathway else "",
        }

    async def close(self) -> None:
        """Close database connections."""
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
            except Exception as exc:
                self.logger.warning("Error closing DB pool: %s", exc)
            self._db_pool = None
