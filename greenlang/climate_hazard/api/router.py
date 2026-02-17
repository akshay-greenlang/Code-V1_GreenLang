# -*- coding: utf-8 -*-
"""
Climate Hazard Connector REST API Router - AGENT-DATA-020

FastAPI router providing 20 REST API endpoints for the Climate
Hazard Connector service at ``/api/v1/climate-hazard``.

Endpoints:
     1. POST   /sources                 - Register hazard source
     2. GET    /sources                 - List sources
     3. GET    /sources/{source_id}     - Get source details
     4. POST   /hazard-data/ingest      - Ingest hazard data
     5. GET    /hazard-data             - Query hazard data
     6. GET    /hazard-data/events      - Historical events
     7. POST   /risk-index/calculate    - Calculate risk index
     8. POST   /risk-index/multi-hazard - Multi-hazard index
     9. POST   /risk-index/compare      - Compare locations
    10. POST   /scenarios/project       - Project scenario
    11. GET    /scenarios               - List scenarios
    12. POST   /assets                  - Register asset
    13. GET    /assets                  - List assets
    14. POST   /exposure/assess         - Assess exposure
    15. POST   /exposure/portfolio      - Portfolio exposure
    16. POST   /vulnerability/score     - Score vulnerability
    17. POST   /reports/generate        - Generate report
    18. GET    /reports/{report_id}     - Get report
    19. POST   /pipeline/run            - Run pipeline
    20. GET    /health                  - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]

router: Optional[Any] = None

if FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/api/v1/climate-hazard", tags=["Climate Hazard"])

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------
    def _get_service():
        from greenlang.climate_hazard.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(status_code=503, detail="Climate Hazard Connector service not initialized")
        return svc

    # 1. POST /sources
    @router.post("/sources")
    async def register_source(body: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new climate hazard data source."""
        svc = _get_service()
        return svc.register_source(**body)

    # 2. GET /sources
    @router.get("/sources")
    async def list_sources(
        hazard_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List registered climate hazard data sources with optional filters."""
        svc = _get_service()
        return svc.list_sources(
            hazard_type=hazard_type, status=status,
            limit=limit, offset=offset,
        )

    # 3. GET /sources/{source_id}
    @router.get("/sources/{source_id}")
    async def get_source(source_id: str) -> Dict[str, Any]:
        """Get climate hazard source details."""
        svc = _get_service()
        result = svc.get_source(source_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
        return result

    # 4. POST /hazard-data/ingest
    @router.post("/hazard-data/ingest")
    async def ingest_hazard_data(body: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest climate hazard data from a registered source."""
        svc = _get_service()
        return svc.ingest_hazard_data(**body)

    # 5. GET /hazard-data
    @router.get("/hazard-data")
    async def query_hazard_data(
        hazard_type: Optional[str] = None,
        source_id: Optional[str] = None,
        location_id: Optional[str] = None,
        scenario: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """Query ingested climate hazard data with optional filters."""
        svc = _get_service()
        return svc.query_hazard_data(
            hazard_type=hazard_type, source_id=source_id,
            location_id=location_id, scenario=scenario,
            limit=limit, offset=offset,
        )

    # 6. GET /hazard-data/events
    @router.get("/hazard-data/events")
    async def list_hazard_events(
        hazard_type: Optional[str] = None,
        severity: Optional[str] = None,
        location_id: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List historical climate hazard events with optional filters."""
        svc = _get_service()
        return svc.list_hazard_events(
            hazard_type=hazard_type, severity=severity,
            location_id=location_id,
            limit=limit, offset=offset,
        )

    # 7. POST /risk-index/calculate
    @router.post("/risk-index/calculate")
    async def calculate_risk_index(body: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate a composite climate risk index for a location."""
        svc = _get_service()
        return svc.calculate_risk_index(**body)

    # 8. POST /risk-index/multi-hazard
    @router.post("/risk-index/multi-hazard")
    async def calculate_multi_hazard(body: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate a multi-hazard composite risk index."""
        svc = _get_service()
        return svc.calculate_multi_hazard(**body)

    # 9. POST /risk-index/compare
    @router.post("/risk-index/compare")
    async def compare_locations(body: Dict[str, Any]) -> Dict[str, Any]:
        """Compare climate risk indices across multiple locations."""
        svc = _get_service()
        return svc.compare_locations(**body)

    # 10. POST /scenarios/project
    @router.post("/scenarios/project")
    async def project_scenario(body: Dict[str, Any]) -> Dict[str, Any]:
        """Project climate hazard under a given SSP or RCP scenario."""
        svc = _get_service()
        return svc.project_scenario(**body)

    # 11. GET /scenarios
    @router.get("/scenarios")
    async def list_scenarios(
        scenario: Optional[str] = None,
        time_horizon: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List available or previously computed scenario projections."""
        svc = _get_service()
        return svc.list_scenarios(
            scenario=scenario, time_horizon=time_horizon,
            limit=limit, offset=offset,
        )

    # 12. POST /assets
    @router.post("/assets")
    async def register_asset(body: Dict[str, Any]) -> Dict[str, Any]:
        """Register a physical or financial asset for climate hazard monitoring."""
        svc = _get_service()
        return svc.register_asset(**body)

    # 13. GET /assets
    @router.get("/assets")
    async def list_assets(
        asset_type: Optional[str] = None,
        location_id: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List registered assets with optional filters."""
        svc = _get_service()
        return svc.list_assets(
            asset_type=asset_type, location_id=location_id,
            limit=limit, offset=offset,
        )

    # 14. POST /exposure/assess
    @router.post("/exposure/assess")
    async def assess_exposure(body: Dict[str, Any]) -> Dict[str, Any]:
        """Assess climate hazard exposure for an asset."""
        svc = _get_service()
        return svc.assess_exposure(**body)

    # 15. POST /exposure/portfolio
    @router.post("/exposure/portfolio")
    async def assess_portfolio_exposure(body: Dict[str, Any]) -> Dict[str, Any]:
        """Assess climate hazard exposure for an entire asset portfolio."""
        svc = _get_service()
        return svc.assess_portfolio_exposure(**body)

    # 16. POST /vulnerability/score
    @router.post("/vulnerability/score")
    async def score_vulnerability(body: Dict[str, Any]) -> Dict[str, Any]:
        """Score climate vulnerability for an asset or entity."""
        svc = _get_service()
        return svc.score_vulnerability(**body)

    # 17. POST /reports/generate
    @router.post("/reports/generate")
    async def generate_report(body: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a climate hazard compliance report (TCFD, CSRD, EU Taxonomy)."""
        svc = _get_service()
        return svc.generate_report(**body)

    # 18. GET /reports/{report_id}
    @router.get("/reports/{report_id}")
    async def get_report(report_id: str) -> Dict[str, Any]:
        """Retrieve a previously generated climate hazard report."""
        svc = _get_service()
        result = svc.get_report(report_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        return result

    # 19. POST /pipeline/run
    @router.post("/pipeline/run")
    async def run_pipeline(body: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full climate hazard assessment pipeline."""
        svc = _get_service()
        return svc.run_pipeline(**body)

    # 20. GET /health
    @router.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check for the climate hazard connector service."""
        svc = _get_service()
        return svc.get_health()


__all__ = ["router"]
