# -*- coding: utf-8 -*-
"""
Deforestation Satellite Connector REST API Router - AGENT-DATA-007: GL-DATA-GEO-003

FastAPI router providing 20 endpoints for satellite imagery acquisition,
vegetation index computation, forest change detection, deforestation alert
integration, EUDR baseline assessment, land cover classification, compliance
report generation, monitoring job management, statistics, and health.

All endpoints are mounted under ``/v1/deforestation``.

Endpoints:
    1.  POST   /v1/deforestation/imagery/acquire              - Acquire satellite imagery
    2.  POST   /v1/deforestation/imagery/time-series           - Acquire time series
    3.  GET    /v1/deforestation/imagery/{scene_id}            - Get scene details
    4.  POST   /v1/deforestation/indices/calculate             - Calculate vegetation indices
    5.  POST   /v1/deforestation/change/detect                 - Detect forest change
    6.  POST   /v1/deforestation/change/trend                  - Analyze NDVI trend
    7.  POST   /v1/deforestation/alerts/query                  - Query deforestation alerts
    8.  POST   /v1/deforestation/alerts/filter-cutoff          - Filter alerts by EUDR cutoff
    9.  POST   /v1/deforestation/baseline/check                - Check baseline (point)
    10. POST   /v1/deforestation/baseline/check-polygon        - Check baseline (polygon)
    11. GET    /v1/deforestation/baseline/forest-definition/{country_iso3} - Get forest def
    12. POST   /v1/deforestation/classify                      - Classify land cover
    13. POST   /v1/deforestation/classify/batch                - Batch classify
    14. POST   /v1/deforestation/compliance/report             - Generate compliance report
    15. GET    /v1/deforestation/compliance/report/{report_id} - Get compliance report
    16. POST   /v1/deforestation/monitoring/start              - Start monitoring job
    17. GET    /v1/deforestation/monitoring/{job_id}            - Get monitoring status
    18. POST   /v1/deforestation/monitoring/{job_id}/stop      - Stop monitoring job
    19. GET    /v1/deforestation/statistics                     - Get service statistics
    20. GET    /v1/deforestation/health                        - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
Status: Production Ready
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import (no `from __future__ import annotations` here
# to avoid issues with FastAPI dependency injection)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; deforestation satellite router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class AcquireImageryBody(BaseModel):
        """Request body for acquiring satellite imagery."""
        polygon_coordinates: List[List[float]] = Field(
            ..., description="List of [lon, lat] coordinate pairs defining the polygon",
        )
        satellite: Optional[str] = Field(
            None, description="Satellite source (sentinel2, landsat8, landsat9, modis)",
        )
        start_date: str = Field(
            ..., description="Start date (ISO YYYY-MM-DD)",
        )
        end_date: str = Field(
            ..., description="End date (ISO YYYY-MM-DD)",
        )
        max_cloud_cover: Optional[int] = Field(
            None, ge=0, le=100, description="Max cloud cover percentage",
        )

    class TimeSeriesBody(BaseModel):
        """Request body for acquiring a time series of imagery."""
        polygon_coordinates: List[List[float]] = Field(
            ..., description="Polygon coordinate pairs",
        )
        satellite: Optional[str] = Field(None, description="Satellite source")
        start_date: str = Field(..., description="Start date")
        end_date: str = Field(..., description="End date")
        max_cloud_cover: Optional[int] = Field(None, ge=0, le=100)
        interval_days: int = Field(default=30, ge=1, description="Days between acquisitions")

    class CalculateIndicesBody(BaseModel):
        """Request body for calculating vegetation indices."""
        scene_id: str = Field(..., description="Scene ID to compute indices for")
        indices: List[str] = Field(
            ..., description="List of index names (ndvi, evi, ndwi, nbr, savi, msavi, ndmi)",
        )

    class DetectChangeBody(BaseModel):
        """Request body for forest change detection."""
        polygon_coordinates: List[List[float]] = Field(
            ..., description="Polygon coordinate pairs",
        )
        pre_start_date: str = Field(..., description="Pre-change start date")
        pre_end_date: str = Field(..., description="Pre-change end date")
        post_start_date: str = Field(..., description="Post-change start date")
        post_end_date: str = Field(..., description="Post-change end date")
        satellite: Optional[str] = Field(None, description="Satellite source")

    class TrendAnalysisBody(BaseModel):
        """Request body for NDVI trend analysis."""
        ndvi_series: List[float] = Field(..., description="NDVI values over time")
        dates: List[str] = Field(..., description="ISO date strings for each value")

    class QueryAlertsBody(BaseModel):
        """Request body for querying deforestation alerts."""
        polygon_coordinates: List[List[float]] = Field(
            ..., description="Polygon coordinate pairs",
        )
        start_date: str = Field(..., description="Query start date")
        end_date: str = Field(..., description="Query end date")
        sources: Optional[List[str]] = Field(
            None, description="Alert sources to query (glad, radd, firms)",
        )
        min_confidence: Optional[str] = Field(
            None, description="Minimum confidence level (low, nominal, high)",
        )

    class FilterCutoffBody(BaseModel):
        """Request body for filtering alerts by EUDR cutoff."""
        alerts: List[Dict[str, Any]] = Field(
            ..., description="List of alert dictionaries to filter",
        )
        cutoff_date: str = Field(
            default="2020-12-31", description="EUDR cutoff date",
        )

    class CheckBaselineBody(BaseModel):
        """Request body for baseline check at a point."""
        latitude: float = Field(..., description="Latitude of the point")
        longitude: float = Field(..., description="Longitude of the point")
        country_iso3: str = Field(..., description="ISO 3166-1 alpha-3 country code")
        observation_date: Optional[str] = Field(
            None, description="Observation date override",
        )

    class CheckBaselinePolygonBody(BaseModel):
        """Request body for baseline check over a polygon."""
        polygon_coordinates: List[List[float]] = Field(
            ..., description="Polygon coordinate pairs",
        )
        country_iso3: str = Field(..., description="ISO 3166-1 alpha-3 country code")
        observation_date: Optional[str] = Field(None, description="Observation date override")
        sample_points: int = Field(default=9, ge=1, description="Sample points for grid")

    class ClassifyBody(BaseModel):
        """Request body for land cover classification."""
        ndvi: float = Field(..., description="NDVI value (-1 to 1)")
        evi: Optional[float] = Field(None, description="EVI value")
        ndwi: Optional[float] = Field(None, description="NDWI value")
        savi: Optional[float] = Field(None, description="SAVI value")

    class ClassifyBatchBody(BaseModel):
        """Request body for batch land cover classification."""
        ndvi_values: List[float] = Field(..., description="List of NDVI values")
        evi_values: Optional[List[float]] = Field(None, description="List of EVI values")
        ndwi_values: Optional[List[float]] = Field(None, description="List of NDWI values")

    class GenerateReportBody(BaseModel):
        """Request body for generating a compliance report."""
        baseline_assessment_id: Optional[str] = Field(
            None, description="Existing baseline assessment ID",
        )
        alert_aggregation_polygon: Optional[List[List[float]]] = Field(
            None, description="Polygon for alert query",
        )
        polygon_wkt: str = Field(..., description="WKT polygon for the report")
        country_iso3: str = Field(..., description="Country code")
        alert_start_date: str = Field(default="2020-01-01", description="Alert query start")
        alert_end_date: str = Field(default="2025-12-31", description="Alert query end")

    class StartMonitoringBody(BaseModel):
        """Request body for starting a monitoring job."""
        polygon_coordinates: List[List[float]] = Field(
            ..., description="Polygon coordinate pairs",
        )
        country_iso3: str = Field(..., description="ISO 3166-1 alpha-3 country code")
        frequency: str = Field(default="monthly", description="Monitoring frequency")
        satellite: Optional[str] = Field(None, description="Satellite source")


# ---------------------------------------------------------------------------
# Router construction
# ---------------------------------------------------------------------------

router = None

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/v1/deforestation",
        tags=["Deforestation Satellite"],
    )

    # ------------------------------------------------------------------
    # Helper: get service from app state
    # ------------------------------------------------------------------

    def _get_service(request: Request) -> Any:
        """Extract DeforestationSatelliteService from request app state.

        Args:
            request: FastAPI Request object.

        Returns:
            DeforestationSatelliteService instance.

        Raises:
            HTTPException: If service is not configured.
        """
        service = getattr(request.app.state, "deforestation_satellite_service", None)
        if service is None:
            raise HTTPException(
                status_code=503,
                detail="Deforestation satellite service not configured",
            )
        return service

    # ==================================================================
    # 1. POST /imagery/acquire - Acquire satellite imagery
    # ==================================================================

    @router.post("/imagery/acquire", summary="Acquire satellite imagery")
    async def acquire_imagery(body: AcquireImageryBody, request: Request) -> Any:
        """Acquire satellite imagery for a polygon area of interest."""
        try:
            service = _get_service(request)
            from greenlang.deforestation_satellite.models import AcquireSatelliteRequest
            req = AcquireSatelliteRequest(
                polygon_coordinates=body.polygon_coordinates,
                satellite=body.satellite,
                start_date=body.start_date,
                end_date=body.end_date,
                max_cloud_cover=body.max_cloud_cover,
            )
            scene = service.acquire_imagery(req)
            return scene.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /imagery/time-series - Acquire time series
    # ==================================================================

    @router.post("/imagery/time-series", summary="Acquire imagery time series")
    async def acquire_time_series(body: TimeSeriesBody, request: Request) -> Any:
        """Acquire multiple scenes over a date range at regular intervals."""
        try:
            service = _get_service(request)
            if service.satellite_engine is None:
                raise HTTPException(status_code=503, detail="Satellite engine not available")
            from greenlang.deforestation_satellite.models import AcquireSatelliteRequest
            req = AcquireSatelliteRequest(
                polygon_coordinates=body.polygon_coordinates,
                satellite=body.satellite,
                start_date=body.start_date,
                end_date=body.end_date,
                max_cloud_cover=body.max_cloud_cover,
            )
            scenes = service.satellite_engine.acquire_time_series(
                req, interval_days=body.interval_days,
            )
            return [s.model_dump(mode="json") for s in scenes]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. GET /imagery/{scene_id} - Get scene details
    # ==================================================================

    @router.get("/imagery/{scene_id}", summary="Get satellite scene details")
    async def get_scene(scene_id: str, request: Request) -> Any:
        """Retrieve metadata and bands for a previously acquired scene."""
        try:
            service = _get_service(request)
            if service.satellite_engine is None:
                raise HTTPException(status_code=503, detail="Satellite engine not available")
            scene = service.satellite_engine.get_scene(scene_id)
            if scene is None:
                raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
            return scene.model_dump(mode="json")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 4. POST /indices/calculate - Calculate vegetation indices
    # ==================================================================

    @router.post("/indices/calculate", summary="Calculate vegetation indices")
    async def calculate_indices(body: CalculateIndicesBody, request: Request) -> Any:
        """Compute vegetation indices for a previously acquired scene."""
        try:
            service = _get_service(request)
            if service.satellite_engine is None:
                raise HTTPException(status_code=503, detail="Satellite engine not available")
            scene = service.satellite_engine.get_scene(body.scene_id)
            if scene is None:
                raise HTTPException(status_code=404, detail=f"Scene {body.scene_id} not found")
            from greenlang.deforestation_satellite.models import VegetationIndex
            index_enums = []
            for name in body.indices:
                try:
                    index_enums.append(VegetationIndex(name.lower()))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown vegetation index: {name}",
                    )
            results = service.satellite_engine.calculate_indices(scene, index_enums)
            return {
                k.value: v.model_dump(mode="json") for k, v in results.items()
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 5. POST /change/detect - Detect forest change
    # ==================================================================

    @router.post("/change/detect", summary="Detect forest change")
    async def detect_change(body: DetectChangeBody, request: Request) -> Any:
        """Perform bi-temporal change detection for a polygon."""
        try:
            service = _get_service(request)
            from greenlang.deforestation_satellite.models import DetectChangeRequest
            req = DetectChangeRequest(
                polygon_coordinates=body.polygon_coordinates,
                pre_start_date=body.pre_start_date,
                pre_end_date=body.pre_end_date,
                post_start_date=body.post_start_date,
                post_end_date=body.post_end_date,
                satellite=body.satellite,
            )
            result = service.detect_change(req)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 6. POST /change/trend - Analyze NDVI trend
    # ==================================================================

    @router.post("/change/trend", summary="Analyze NDVI trend")
    async def analyze_trend(body: TrendAnalysisBody, request: Request) -> Any:
        """Perform linear regression trend analysis on NDVI time series."""
        try:
            service = _get_service(request)
            if service.change_engine is None:
                raise HTTPException(status_code=503, detail="Change engine not available")
            trend = service.change_engine.analyze_trend(body.ndvi_series, body.dates)
            return trend.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 7. POST /alerts/query - Query deforestation alerts
    # ==================================================================

    @router.post("/alerts/query", summary="Query deforestation alerts")
    async def query_alerts(body: QueryAlertsBody, request: Request) -> Any:
        """Query deforestation alerts for a polygon from multiple sources."""
        try:
            service = _get_service(request)
            from greenlang.deforestation_satellite.models import QueryAlertsRequest
            req = QueryAlertsRequest(
                polygon_coordinates=body.polygon_coordinates,
                start_date=body.start_date,
                end_date=body.end_date,
                sources=body.sources,
                min_confidence=body.min_confidence,
            )
            aggregation = service.query_alerts(req)
            return aggregation.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 8. POST /alerts/filter-cutoff - Filter alerts by EUDR cutoff
    # ==================================================================

    @router.post("/alerts/filter-cutoff", summary="Filter alerts by EUDR cutoff")
    async def filter_cutoff(body: FilterCutoffBody, request: Request) -> Any:
        """Filter a list of alerts to only include post-EUDR-cutoff events."""
        try:
            service = _get_service(request)
            if service.alert_engine is None:
                raise HTTPException(status_code=503, detail="Alert engine not available")
            from greenlang.deforestation_satellite.models import DeforestationAlert
            alerts = [DeforestationAlert(**a) for a in body.alerts]
            filtered = service.alert_engine.filter_post_cutoff(alerts, body.cutoff_date)
            return [a.model_dump(mode="json") for a in filtered]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. POST /baseline/check - Check baseline (point)
    # ==================================================================

    @router.post("/baseline/check", summary="Check EUDR baseline (point)")
    async def check_baseline(body: CheckBaselineBody, request: Request) -> Any:
        """Check EUDR baseline compliance for a single coordinate."""
        try:
            service = _get_service(request)
            from greenlang.deforestation_satellite.models import CheckBaselineRequest
            req = CheckBaselineRequest(
                latitude=body.latitude,
                longitude=body.longitude,
                country_iso3=body.country_iso3,
                observation_date=body.observation_date,
            )
            assessment = service.check_baseline(req)
            return assessment.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 10. POST /baseline/check-polygon - Check baseline (polygon)
    # ==================================================================

    @router.post("/baseline/check-polygon", summary="Check EUDR baseline (polygon)")
    async def check_baseline_polygon(body: CheckBaselinePolygonBody, request: Request) -> Any:
        """Check EUDR baseline compliance across a polygon area."""
        try:
            service = _get_service(request)
            from greenlang.deforestation_satellite.models import CheckBaselinePolygonRequest
            req = CheckBaselinePolygonRequest(
                polygon_coordinates=body.polygon_coordinates,
                country_iso3=body.country_iso3,
                observation_date=body.observation_date,
                sample_points=body.sample_points,
            )
            assessment = service.check_baseline_polygon(req)
            return assessment.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 11. GET /baseline/forest-definition/{country_iso3}
    # ==================================================================

    @router.get(
        "/baseline/forest-definition/{country_iso3}",
        summary="Get forest definition for country",
    )
    async def get_forest_definition(country_iso3: str, request: Request) -> Any:
        """Retrieve the forest definition parameters for a country."""
        try:
            service = _get_service(request)
            if service.baseline_engine is None:
                raise HTTPException(status_code=503, detail="Baseline engine not available")
            definition = service.baseline_engine.get_forest_definition(country_iso3)
            return definition.model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. POST /classify - Classify land cover
    # ==================================================================

    @router.post("/classify", summary="Classify land cover")
    async def classify_land_cover(body: ClassifyBody, request: Request) -> Any:
        """Classify land cover type from vegetation indices."""
        try:
            service = _get_service(request)
            classification = service.classify_land_cover(
                ndvi=body.ndvi, evi=body.evi, ndwi=body.ndwi, savi=body.savi,
            )
            return classification.model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 13. POST /classify/batch - Batch classify
    # ==================================================================

    @router.post("/classify/batch", summary="Batch classify land cover")
    async def classify_batch(body: ClassifyBatchBody, request: Request) -> Any:
        """Classify land cover for multiple pixels in batch."""
        try:
            service = _get_service(request)
            if service.classifier_engine is None:
                raise HTTPException(status_code=503, detail="Classifier engine not available")
            results = service.classifier_engine.classify_batch(
                ndvi_values=body.ndvi_values,
                evi_values=body.evi_values,
                ndwi_values=body.ndwi_values,
            )
            return [r.model_dump(mode="json") for r in results]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 14. POST /compliance/report - Generate compliance report
    # ==================================================================

    @router.post("/compliance/report", summary="Generate EUDR compliance report")
    async def generate_compliance_report(body: GenerateReportBody, request: Request) -> Any:
        """Generate a full EUDR compliance report for a polygon."""
        try:
            service = _get_service(request)

            # Get or create baseline assessment
            if service.baseline_engine is None:
                raise HTTPException(status_code=503, detail="Baseline engine not available")

            polygon_coords = body.alert_aggregation_polygon or []
            if polygon_coords:
                from greenlang.deforestation_satellite.models import CheckBaselinePolygonRequest
                baseline_req = CheckBaselinePolygonRequest(
                    polygon_coordinates=polygon_coords,
                    country_iso3=body.country_iso3,
                )
                baseline = service.baseline_engine.check_baseline_polygon(baseline_req)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="alert_aggregation_polygon is required to generate a report",
                )

            # Query alerts
            if service.alert_engine is not None and polygon_coords:
                from greenlang.deforestation_satellite.models import QueryAlertsRequest
                alert_req = QueryAlertsRequest(
                    polygon_coordinates=polygon_coords,
                    start_date=body.alert_start_date,
                    end_date=body.alert_end_date,
                )
                alerts = service.alert_engine.query_alerts(alert_req)
            else:
                from greenlang.deforestation_satellite.models import AlertAggregation
                alerts = AlertAggregation(polygon_wkt=body.polygon_wkt)

            # Generate report
            report = service.generate_compliance_report(
                baseline=baseline,
                alerts=alerts,
                polygon_wkt=body.polygon_wkt,
            )
            return report.model_dump(mode="json")
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 15. GET /compliance/report/{report_id} - Get compliance report
    # ==================================================================

    @router.get("/compliance/report/{report_id}", summary="Get compliance report")
    async def get_compliance_report(report_id: str, request: Request) -> Any:
        """Retrieve a previously generated compliance report."""
        try:
            service = _get_service(request)
            if service.report_engine is None:
                raise HTTPException(status_code=503, detail="Report engine not available")
            report = service.report_engine.get_report(report_id)
            if report is None:
                raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
            return report.model_dump(mode="json")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. POST /monitoring/start - Start monitoring job
    # ==================================================================

    @router.post("/monitoring/start", summary="Start monitoring job")
    async def start_monitoring(body: StartMonitoringBody, request: Request) -> Any:
        """Start a deforestation monitoring pipeline job."""
        try:
            service = _get_service(request)
            from greenlang.deforestation_satellite.models import StartMonitoringRequest
            req = StartMonitoringRequest(
                polygon_coordinates=body.polygon_coordinates,
                country_iso3=body.country_iso3,
                frequency=body.frequency,
                satellite=body.satellite,
            )
            job = service.start_monitoring(req)
            return job.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 17. GET /monitoring/{job_id} - Get monitoring status
    # ==================================================================

    @router.get("/monitoring/{job_id}", summary="Get monitoring job status")
    async def get_monitoring_status(job_id: str, request: Request) -> Any:
        """Retrieve the status of a monitoring job."""
        try:
            service = _get_service(request)
            job = service.get_monitoring_status(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            return job.model_dump(mode="json")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 18. POST /monitoring/{job_id}/stop - Stop monitoring job
    # ==================================================================

    @router.post("/monitoring/{job_id}/stop", summary="Stop monitoring job")
    async def stop_monitoring(job_id: str, request: Request) -> Any:
        """Stop a running monitoring job."""
        try:
            service = _get_service(request)
            result = service.stop_monitoring(job_id)
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Job {job_id} not found or not running",
                )
            return {"job_id": job_id, "stopped": True}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 19. GET /statistics - Get service statistics
    # ==================================================================

    @router.get("/statistics", summary="Get service statistics")
    async def get_statistics(request: Request) -> Any:
        """Retrieve aggregate deforestation satellite service statistics."""
        try:
            service = _get_service(request)
            stats = service.get_statistics()
            return stats.model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 20. GET /health - Health check
    # ==================================================================

    @router.get("/health", summary="Health check")
    async def health_check(request: Request) -> Any:
        """Health check endpoint for the deforestation satellite service."""
        try:
            service = _get_service(request)
            return {
                "status": "healthy",
                "service": "deforestation_satellite_connector",
                "version": "1.0.0",
                "agent": "AGENT-DATA-007",
                "component": "GL-DATA-GEO-003",
                "started": service._started,
                "engines": {
                    "satellite_data": service.satellite_engine is not None,
                    "forest_change": service.change_engine is not None,
                    "alert_aggregation": service.alert_engine is not None,
                    "baseline_assessment": service.baseline_engine is not None,
                    "classifier": service.classifier_engine is not None,
                    "compliance_report": service.report_engine is not None,
                    "monitoring_pipeline": service.pipeline_engine is not None,
                },
                "provenance_entries": (
                    service.provenance.entry_count
                    if service.provenance else 0
                ),
            }
        except Exception:
            return {
                "status": "degraded",
                "service": "deforestation_satellite_connector",
                "version": "1.0.0",
                "agent": "AGENT-DATA-007",
                "component": "GL-DATA-GEO-003",
                "started": False,
            }


__all__ = [
    "router",
]
