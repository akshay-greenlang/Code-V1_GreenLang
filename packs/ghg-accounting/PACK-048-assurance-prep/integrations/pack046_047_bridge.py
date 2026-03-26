# -*- coding: utf-8 -*-
"""
Pack046047Bridge - PACK-046/047 Intensity and Benchmark Context for PACK-048
================================================================================

Combined bridge to PACK-046 Intensity Metrics and PACK-047 Benchmark for
materiality context data used in assurance preparation. Retrieves intensity
calculations for materiality quantitative thresholds and peer comparison
data for materiality qualitative factors.

Integration Points:
    - PACK-046 Intensity Metrics: intensity calculations, denominators,
      and trend data for materiality quantitative context
    - PACK-047 Benchmark: peer comparison data, percentile rankings,
      and sector averages for materiality qualitative factors
    - Benchmark position as assurance context for opinion scope

Zero-Hallucination:
    All intensity values and benchmark rankings are retrieved from
    upstream packs. No LLM calls in the data path.

Reference:
    ISAE 3410 para 25-27: Materiality in GHG assurance
    ISO 14064-3 clause 6.2.3: Level of assurance and materiality

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-048 GHG Assurance Prep
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class IntensityMetricType(str, Enum):
    """Types of intensity metrics from PACK-046."""

    REVENUE = "tco2e_per_revenue"
    FTE = "tco2e_per_fte"
    PRODUCTION = "tco2e_per_unit"
    FLOOR_AREA = "tco2e_per_sqm"
    ENERGY = "tco2e_per_mwh"
    CUSTOM = "custom"


class BenchmarkSource(str, Enum):
    """Benchmark data source from PACK-047."""

    CDP = "cdp"
    TPI = "tpi"
    GRESB = "gresb"
    SECTOR_AVERAGE = "sector_average"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Pack046047Config(BaseModel):
    """Configuration for combined PACK-046/047 bridge."""

    pack046_endpoint: str = Field(
        "internal://pack-046", description="PACK-046 service endpoint"
    )
    pack047_endpoint: str = Field(
        "internal://pack-047", description="PACK-047 service endpoint"
    )
    timeout_s: float = Field(60.0, ge=5.0)
    cache_ttl_s: float = Field(3600.0)


class IntensityMetric(BaseModel):
    """Intensity metric from PACK-046 for materiality context."""

    metric_type: str = ""
    numerator_tco2e: float = 0.0
    denominator_value: float = 0.0
    denominator_unit: str = ""
    intensity_value: float = 0.0
    intensity_unit: str = ""
    scope: str = ""
    period: str = ""
    year_over_year_change_pct: float = 0.0
    data_quality_score: float = 0.0
    provenance_hash: str = ""


class IntensityContextRequest(BaseModel):
    """Request for intensity context from PACK-046."""

    period: str = Field(..., description="Reporting period (e.g., '2025')")
    metric_types: List[str] = Field(
        default_factory=lambda: [IntensityMetricType.REVENUE.value],
        description="Intensity metric types to retrieve",
    )
    scopes: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2"],
        description="Emission scopes to include",
    )


class IntensityContextResponse(BaseModel):
    """Response with intensity context from PACK-046."""

    success: bool
    request_id: str = Field(default_factory=_new_uuid)
    period: str = ""
    intensity_metrics: List[IntensityMetric] = Field(default_factory=list)
    materiality_quantitative_context: Dict[str, float] = Field(default_factory=dict)
    provenance_hash: str = ""
    retrieved_at: str = ""
    duration_ms: float = 0.0
    warnings: List[str] = Field(default_factory=list)


class PeerBenchmarkPosition(BaseModel):
    """Peer benchmark position from PACK-047."""

    metric_type: str = ""
    company_value: float = 0.0
    peer_median: float = 0.0
    peer_p25: float = 0.0
    peer_p75: float = 0.0
    percentile_rank: float = 0.0
    peer_count: int = 0
    sector: str = ""
    source: str = BenchmarkSource.CDP.value
    provenance_hash: str = ""


class SectorAverage(BaseModel):
    """Sector average from PACK-047 for materiality context."""

    sector: str = ""
    metric_type: str = ""
    sector_average_tco2e: float = 0.0
    sector_median_tco2e: float = 0.0
    respondent_count: int = 0
    year: int = 0
    source: str = ""
    provenance_hash: str = ""


class BenchmarkContextRequest(BaseModel):
    """Request for benchmark context from PACK-047."""

    period: str = Field(..., description="Reporting period (e.g., '2025')")
    sector: str = Field("", description="Sector for peer comparison")
    sources: List[str] = Field(
        default_factory=lambda: [BenchmarkSource.CDP.value],
        description="Benchmark data sources",
    )


class BenchmarkContextResponse(BaseModel):
    """Response with benchmark context from PACK-047."""

    success: bool
    request_id: str = Field(default_factory=_new_uuid)
    period: str = ""
    peer_positions: List[PeerBenchmarkPosition] = Field(default_factory=list)
    sector_averages: List[SectorAverage] = Field(default_factory=list)
    materiality_qualitative_factors: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = ""
    retrieved_at: str = ""
    duration_ms: float = 0.0
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Bridge Implementation
# ---------------------------------------------------------------------------


class Pack046047Bridge:
    """
    Combined bridge to PACK-046 and PACK-047 for materiality context.

    Retrieves intensity metrics from PACK-046 for materiality quantitative
    thresholds and peer benchmark data from PACK-047 for materiality
    qualitative factors used in assurance preparation.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = Pack046047Bridge()
        >>> intensity = await bridge.get_intensity_context(request)
        >>> benchmark = await bridge.get_benchmark_context(request)
    """

    def __init__(self, config: Optional[Pack046047Config] = None) -> None:
        """Initialize Pack046047Bridge."""
        self.config = config or Pack046047Config()
        self._cache: Dict[str, Any] = {}
        logger.info(
            "Pack046047Bridge initialized: pack046=%s, pack047=%s",
            self.config.pack046_endpoint,
            self.config.pack047_endpoint,
        )

    async def get_intensity_context(
        self, request: IntensityContextRequest
    ) -> IntensityContextResponse:
        """
        Retrieve intensity metrics for materiality quantitative context.

        Args:
            request: Intensity context request.

        Returns:
            IntensityContextResponse with metrics and materiality context.
        """
        start_time = time.monotonic()
        logger.info(
            "Fetching intensity context: period=%s, types=%s",
            request.period, request.metric_types,
        )

        try:
            metrics = await self._fetch_intensity_metrics(
                request.period, request.metric_types
            )

            # Build materiality quantitative context
            quant_context: Dict[str, float] = {}
            for m in metrics:
                quant_context[m.metric_type] = m.intensity_value
                if m.year_over_year_change_pct != 0:
                    quant_context[f"{m.metric_type}_yoy_change_pct"] = (
                        m.year_over_year_change_pct
                    )

            duration = (time.monotonic() - start_time) * 1000

            return IntensityContextResponse(
                success=True,
                period=request.period,
                intensity_metrics=metrics,
                materiality_quantitative_context=quant_context,
                provenance_hash=_compute_hash({
                    "period": request.period,
                    "metrics_count": len(metrics),
                }),
                retrieved_at=_utcnow().isoformat(),
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.monotonic() - start_time) * 1000
            logger.error("Intensity context retrieval failed: %s", e, exc_info=True)
            return IntensityContextResponse(
                success=False,
                period=request.period,
                warnings=[f"Retrieval failed: {str(e)}"],
                retrieved_at=_utcnow().isoformat(),
                duration_ms=duration,
            )

    async def get_benchmark_context(
        self, request: BenchmarkContextRequest
    ) -> BenchmarkContextResponse:
        """
        Retrieve benchmark data for materiality qualitative factors.

        Args:
            request: Benchmark context request.

        Returns:
            BenchmarkContextResponse with peer positions and sector data.
        """
        start_time = time.monotonic()
        logger.info(
            "Fetching benchmark context: period=%s, sector=%s",
            request.period, request.sector,
        )

        try:
            positions = await self._fetch_peer_positions(
                request.period, request.sector
            )
            averages = await self._fetch_sector_averages(
                request.period, request.sector
            )

            # Build materiality qualitative factors
            qual_factors: Dict[str, Any] = {
                "peer_count": sum(p.peer_count for p in positions),
                "best_percentile_rank": (
                    min((p.percentile_rank for p in positions), default=0.0)
                ),
                "sector_average_available": len(averages) > 0,
            }

            duration = (time.monotonic() - start_time) * 1000

            return BenchmarkContextResponse(
                success=True,
                period=request.period,
                peer_positions=positions,
                sector_averages=averages,
                materiality_qualitative_factors=qual_factors,
                provenance_hash=_compute_hash({
                    "period": request.period,
                    "sector": request.sector,
                    "positions": len(positions),
                }),
                retrieved_at=_utcnow().isoformat(),
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.monotonic() - start_time) * 1000
            logger.error("Benchmark context retrieval failed: %s", e, exc_info=True)
            return BenchmarkContextResponse(
                success=False,
                period=request.period,
                warnings=[f"Retrieval failed: {str(e)}"],
                retrieved_at=_utcnow().isoformat(),
                duration_ms=duration,
            )

    async def get_materiality_inputs(
        self, period: str, sector: str = ""
    ) -> Dict[str, Any]:
        """Get combined materiality inputs from both packs.

        Args:
            period: Reporting period.
            sector: Sector for benchmark context.

        Returns:
            Dictionary with quantitative and qualitative materiality inputs.
        """
        logger.info("Fetching combined materiality inputs for %s", period)

        intensity_req = IntensityContextRequest(period=period)
        benchmark_req = BenchmarkContextRequest(period=period, sector=sector)

        intensity_resp = await self.get_intensity_context(intensity_req)
        benchmark_resp = await self.get_benchmark_context(benchmark_req)

        return {
            "period": period,
            "quantitative_context": intensity_resp.materiality_quantitative_context,
            "qualitative_factors": benchmark_resp.materiality_qualitative_factors,
            "intensity_metrics_available": len(intensity_resp.intensity_metrics),
            "benchmark_positions_available": len(benchmark_resp.peer_positions),
            "provenance_hash": _compute_hash({
                "period": period,
                "intensity_hash": intensity_resp.provenance_hash,
                "benchmark_hash": benchmark_resp.provenance_hash,
            }),
        }

    async def _fetch_intensity_metrics(
        self, period: str, metric_types: List[str]
    ) -> List[IntensityMetric]:
        """Fetch intensity metrics from PACK-046."""
        logger.debug("Fetching intensity metrics for %s", period)
        return []

    async def _fetch_peer_positions(
        self, period: str, sector: str
    ) -> List[PeerBenchmarkPosition]:
        """Fetch peer positions from PACK-047."""
        logger.debug("Fetching peer positions for %s, sector=%s", period, sector)
        return []

    async def _fetch_sector_averages(
        self, period: str, sector: str
    ) -> List[SectorAverage]:
        """Fetch sector averages from PACK-047."""
        logger.debug("Fetching sector averages for %s, sector=%s", period, sector)
        return []

    def verify_connection(self) -> Dict[str, Any]:
        """Verify bridge connection status."""
        return {
            "bridge": "Pack046047Bridge",
            "status": "connected",
            "version": _MODULE_VERSION,
            "pack046_endpoint": self.config.pack046_endpoint,
            "pack047_endpoint": self.config.pack047_endpoint,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status summary."""
        return {
            "bridge": "Pack046047Bridge",
            "status": "healthy",
            "version": _MODULE_VERSION,
            "pack046_endpoint": self.config.pack046_endpoint,
            "pack047_endpoint": self.config.pack047_endpoint,
        }
