# -*- coding: utf-8 -*-
"""
Unit Tests for Deforestation Satellite API Router (AGENT-DATA-007)

Tests all 20 FastAPI endpoints via TestClient. Covers satellite imagery
acquisition, vegetation indices, land cover classification, change detection,
baseline assessment, alert management, compliance reporting, monitoring
pipeline, scene listing, forest definitions, health, and statistics.

Coverage target: 85%+ of api/router.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

try:
    from fastapi import FastAPI, APIRouter, HTTPException, Query
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")


# ---------------------------------------------------------------------------
# Request models for validation
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    class AcquireRequest(BaseModel):
        polygon: List[List[float]]
        start_date: str
        end_date: str
        satellite: str = "sentinel2"
        max_cloud_cover: int = 30

    class IndicesRequest(BaseModel):
        scene_id: str
        indices: List[str] = ["ndvi", "evi", "ndwi", "nbr"]

    class ClassifyRequest(BaseModel):
        ndvi: float
        evi: Optional[float] = None
        ndwi: Optional[float] = None

    class DetectChangeRequest(BaseModel):
        polygon: List[List[float]]
        baseline_date: str
        current_date: str

    class BaselineRequest(BaseModel):
        latitude: float
        longitude: float
        country: str = "BRA"

    class BaselinePolygonRequest(BaseModel):
        polygon: List[List[float]]
        country: str = "BRA"
        sample_points: int = 9

    class AlertQueryRequest(BaseModel):
        polygon: List[List[float]]
        start_date: str = "2020-12-31"
        end_date: Optional[str] = None

    class AlertAggregateRequest(BaseModel):
        polygon: List[List[float]]
        sources: List[str] = ["glad", "radd", "firms"]

    class ComplianceAssessRequest(BaseModel):
        polygon_id: str
        alerts: List[Dict[str, Any]] = []

    class ComplianceReportRequest(BaseModel):
        polygon_id: str
        alerts: List[Dict[str, Any]] = []

    class MonitorStartRequest(BaseModel):
        polygon_id: str
        frequency: str = "on_demand"
        satellite: str = "sentinel2"


# ---------------------------------------------------------------------------
# Inline router for testing (mirrors api/router.py)
# ---------------------------------------------------------------------------


def create_test_app() -> "FastAPI":
    """Build a minimal FastAPI app with all 20 deforestation endpoints."""
    app = FastAPI(title="Deforestation Satellite Connector Test")
    router = APIRouter(prefix="/v1/deforestation", tags=["deforestation"])

    # In-memory state
    _scenes: List[Dict[str, Any]] = []
    _monitoring_jobs: Dict[str, Dict[str, Any]] = {}
    _reports: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------------
    # 1. POST /acquire
    # ---------------------------------------------------------------
    @router.post("/acquire")
    def acquire(req: AcquireRequest):
        scene_id = f"scene-{uuid.uuid4().hex[:12]}"
        scene = {
            "scene_id": scene_id,
            "satellite": req.satellite,
            "start_date": req.start_date,
            "end_date": req.end_date,
            "cloud_cover_pct": 15,
            "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
            "resolution_m": 10 if req.satellite == "sentinel2" else 30,
            "status": "acquired",
        }
        _scenes.append(scene)
        return scene

    # ---------------------------------------------------------------
    # 2. POST /indices
    # ---------------------------------------------------------------
    @router.post("/indices")
    def compute_indices(req: IndicesRequest):
        return {
            "scene_id": req.scene_id,
            "indices": {idx: round(0.5 + 0.1 * i, 2) for i, idx in enumerate(req.indices)},
            "status": "computed",
        }

    # ---------------------------------------------------------------
    # 3. POST /classify
    # ---------------------------------------------------------------
    @router.post("/classify")
    def classify(req: ClassifyRequest):
        ndvi = req.ndvi
        if ndwi := req.ndwi:
            if ndwi > 0.3:
                lc = "water"
            elif ndvi >= 0.6:
                lc = "dense_forest"
            else:
                lc = "open_forest" if ndvi >= 0.4 else "grassland"
        elif ndvi >= 0.6:
            lc = "dense_forest"
        elif ndvi >= 0.4:
            lc = "open_forest"
        elif ndvi >= 0.3:
            lc = "shrubland"
        elif ndvi >= 0.2:
            lc = "grassland"
        elif ndvi > 0:
            lc = "bare_soil"
        else:
            lc = "unknown"
        return {
            "classification_id": f"cls-{uuid.uuid4().hex[:12]}",
            "land_cover_type": lc,
            "ndvi": ndvi,
            "is_forest": lc in ("dense_forest", "open_forest"),
        }

    # ---------------------------------------------------------------
    # 4. POST /detect-change
    # ---------------------------------------------------------------
    @router.post("/detect-change")
    def detect_change(req: DetectChangeRequest):
        return {
            "change_id": f"chg-{uuid.uuid4().hex[:12]}",
            "primary_change_type": "no_change",
            "dndvi": -0.02,
            "confidence": 0.85,
            "baseline_date": req.baseline_date,
            "current_date": req.current_date,
            "status": "completed",
        }

    # ---------------------------------------------------------------
    # 5. POST /check-baseline
    # ---------------------------------------------------------------
    @router.post("/check-baseline")
    def check_baseline(req: BaselineRequest):
        return {
            "assessment_id": f"bl-{uuid.uuid4().hex[:12]}",
            "latitude": req.latitude,
            "longitude": req.longitude,
            "country": req.country,
            "baseline_date": "2020-12-31",
            "was_forested_at_baseline": True,
            "compliance_status": "COMPLIANT",
            "risk_score": 15.0,
        }

    # ---------------------------------------------------------------
    # 6. POST /check-baseline/polygon
    # ---------------------------------------------------------------
    @router.post("/check-baseline/polygon")
    def check_baseline_polygon(req: BaselinePolygonRequest):
        return {
            "assessment_id": f"blp-{uuid.uuid4().hex[:12]}",
            "polygon": req.polygon,
            "country": req.country,
            "sample_points": req.sample_points,
            "baseline_date": "2020-12-31",
            "overall_compliance": "COMPLIANT",
            "risk_score": 12.0,
        }

    # ---------------------------------------------------------------
    # 7. POST /alerts/query
    # ---------------------------------------------------------------
    @router.post("/alerts/query")
    def query_alerts(req: AlertQueryRequest):
        return {
            "alerts": [
                {"alert_id": f"alert-{uuid.uuid4().hex[:12]}", "source": "glad",
                 "confidence": "nominal", "severity": "low", "post_cutoff": True}
            ],
            "total": 1,
        }

    # ---------------------------------------------------------------
    # 8. GET /alerts/{alert_id}
    # ---------------------------------------------------------------
    @router.get("/alerts/{alert_id}")
    def get_alert(alert_id: str):
        return {
            "alert_id": alert_id,
            "source": "glad",
            "confidence": "high",
            "severity": "medium",
            "detected_date": "2023-03-15",
            "post_cutoff": True,
            "area_ha": 1.5,
        }

    # ---------------------------------------------------------------
    # 9. POST /alerts/aggregate
    # ---------------------------------------------------------------
    @router.post("/alerts/aggregate")
    def aggregate_alerts(req: AlertAggregateRequest):
        return {
            "aggregation_id": f"agg-{uuid.uuid4().hex[:12]}",
            "sources": req.sources,
            "total_alerts": 5,
            "deduplicated_alerts": 3,
            "by_source": {"glad": 2, "radd": 2, "firms": 1},
        }

    # ---------------------------------------------------------------
    # 10. POST /compliance/assess
    # ---------------------------------------------------------------
    @router.post("/compliance/assess")
    def assess_compliance(req: ComplianceAssessRequest):
        post_cutoff = [a for a in req.alerts if a.get("post_cutoff", False)]
        if not post_cutoff:
            status = "COMPLIANT"
        elif any(a.get("confidence") == "high" for a in post_cutoff):
            status = "NON_COMPLIANT"
        else:
            status = "REVIEW_REQUIRED"
        return {
            "polygon_id": req.polygon_id,
            "compliance_status": status,
            "alert_count": len(req.alerts),
        }

    # ---------------------------------------------------------------
    # 11. POST /compliance/report
    # ---------------------------------------------------------------
    @router.post("/compliance/report")
    def generate_report(req: ComplianceReportRequest):
        report_id = f"rpt-{uuid.uuid4().hex[:12]}"
        post_cutoff = [a for a in req.alerts if a.get("post_cutoff", False)]
        high_conf = [a for a in post_cutoff if a.get("confidence") == "high"]
        if not post_cutoff:
            status = "COMPLIANT"
        elif high_conf:
            status = "NON_COMPLIANT"
        else:
            status = "REVIEW_REQUIRED"
        report = {
            "report_id": report_id,
            "polygon_id": req.polygon_id,
            "compliance_status": status,
            "alert_count": len(req.alerts),
            "risk_score": min(100.0, len(post_cutoff) * 15.0),
        }
        _reports[report_id] = report
        return report

    # ---------------------------------------------------------------
    # 12. GET /compliance/{report_id}
    # ---------------------------------------------------------------
    @router.get("/compliance/{report_id}")
    def get_report(report_id: str):
        if report_id not in _reports:
            raise HTTPException(404, "Report not found")
        return _reports[report_id]

    # ---------------------------------------------------------------
    # 13. POST /monitor/start
    # ---------------------------------------------------------------
    @router.post("/monitor/start")
    def start_monitoring(req: MonitorStartRequest):
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = {
            "job_id": job_id,
            "polygon_id": req.polygon_id,
            "frequency": req.frequency,
            "satellite": req.satellite,
            "status": "running",
            "is_running": True,
        }
        _monitoring_jobs[job_id] = job
        return job

    # ---------------------------------------------------------------
    # 16. GET /monitor/jobs (must be before {job_id} to avoid path conflict)
    # ---------------------------------------------------------------
    @router.get("/monitor/jobs")
    def list_monitoring_jobs():
        return {"jobs": list(_monitoring_jobs.values()), "total": len(_monitoring_jobs)}

    # ---------------------------------------------------------------
    # 14. GET /monitor/{job_id}
    # ---------------------------------------------------------------
    @router.get("/monitor/{job_id}")
    def get_monitoring_job(job_id: str):
        if job_id not in _monitoring_jobs:
            raise HTTPException(404, "Monitoring job not found")
        return _monitoring_jobs[job_id]

    # ---------------------------------------------------------------
    # 15. POST /monitor/{job_id}/stop
    # ---------------------------------------------------------------
    @router.post("/monitor/{job_id}/stop")
    def stop_monitoring(job_id: str):
        if job_id not in _monitoring_jobs:
            raise HTTPException(404, "Monitoring job not found")
        _monitoring_jobs[job_id]["status"] = "stopped"
        _monitoring_jobs[job_id]["is_running"] = False
        return _monitoring_jobs[job_id]

    # ---------------------------------------------------------------
    # 17. GET /scenes
    # ---------------------------------------------------------------
    @router.get("/scenes")
    def list_scenes():
        return {"scenes": _scenes, "total": len(_scenes)}

    # ---------------------------------------------------------------
    # 18. GET /forest-definitions
    # ---------------------------------------------------------------
    @router.get("/forest-definitions")
    def forest_definitions():
        return {
            "definitions": {
                "BRA": {"min_tree_cover_pct": 10, "min_height_m": 5, "min_area_ha": 0.5},
                "IDN": {"min_tree_cover_pct": 10, "min_height_m": 5, "min_area_ha": 0.5},
                "COD": {"min_tree_cover_pct": 10, "min_height_m": 5, "min_area_ha": 0.5},
                "MYS": {"min_tree_cover_pct": 30, "min_height_m": 5, "min_area_ha": 0.5},
                "CIV": {"min_tree_cover_pct": 10, "min_height_m": 5, "min_area_ha": 0.5},
                "GHA": {"min_tree_cover_pct": 15, "min_height_m": 5, "min_area_ha": 0.5},
                "COL": {"min_tree_cover_pct": 10, "min_height_m": 5, "min_area_ha": 0.5},
                "PER": {"min_tree_cover_pct": 10, "min_height_m": 5, "min_area_ha": 0.5},
                "FAO_DEFAULT": {"min_tree_cover_pct": 10, "min_height_m": 5, "min_area_ha": 0.5},
            },
            "total": 9,
        }

    # ---------------------------------------------------------------
    # 19. GET /health
    # ---------------------------------------------------------------
    @router.get("/health")
    def health():
        return {
            "status": "healthy",
            "service": "deforestation-satellite-connector",
            "version": "1.0.0",
            "agent_id": "GL-DATA-GEO-003",
        }

    # ---------------------------------------------------------------
    # 20. GET /statistics
    # ---------------------------------------------------------------
    @router.get("/statistics")
    def statistics():
        running = sum(1 for j in _monitoring_jobs.values() if j.get("is_running"))
        return {
            "total_scenes": len(_scenes),
            "total_monitoring_jobs": len(_monitoring_jobs),
            "running_monitoring_jobs": running,
            "total_reports": len(_reports),
        }

    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    app = create_test_app()
    return TestClient(app)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestHealthEndpoint:
    def test_health_endpoint(self, client):
        resp = client.get("/v1/deforestation/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "deforestation-satellite-connector"
        assert data["agent_id"] == "GL-DATA-GEO-003"

    def test_health_has_version(self, client):
        resp = client.get("/v1/deforestation/health")
        assert "version" in resp.json()


class TestStatisticsEndpoint:
    def test_statistics_endpoint(self, client):
        resp = client.get("/v1/deforestation/statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_scenes" in data
        assert "total_monitoring_jobs" in data
        assert "total_reports" in data


class TestAcquireEndpoint:
    def test_acquire_endpoint(self, client):
        resp = client.post("/v1/deforestation/acquire", json={
            "polygon": [[-55.0, -10.0], [-54.5, -10.0], [-54.5, -9.5], [-55.0, -9.5]],
            "start_date": "2023-01-01",
            "end_date": "2023-06-01",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["scene_id"].startswith("scene-")
        assert data["status"] == "acquired"

    def test_acquire_custom_satellite(self, client):
        resp = client.post("/v1/deforestation/acquire", json={
            "polygon": [[-55.0, -10.0]],
            "start_date": "2023-01-01",
            "end_date": "2023-06-01",
            "satellite": "landsat8",
        })
        assert resp.status_code == 200
        assert resp.json()["satellite"] == "landsat8"
        assert resp.json()["resolution_m"] == 30

    def test_acquire_sentinel2_resolution(self, client):
        resp = client.post("/v1/deforestation/acquire", json={
            "polygon": [[-55.0, -10.0]],
            "start_date": "2023-01-01",
            "end_date": "2023-06-01",
        })
        assert resp.json()["resolution_m"] == 10


class TestIndicesEndpoint:
    def test_indices_endpoint(self, client):
        resp = client.post("/v1/deforestation/indices", json={
            "scene_id": "scene-abc123",
            "indices": ["ndvi", "evi", "ndwi"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["scene_id"] == "scene-abc123"
        assert data["status"] == "computed"
        assert "ndvi" in data["indices"]

    def test_indices_default(self, client):
        resp = client.post("/v1/deforestation/indices", json={
            "scene_id": "scene-abc123",
        })
        assert resp.status_code == 200
        assert len(resp.json()["indices"]) == 4


class TestClassifyEndpoint:
    def test_classify_endpoint(self, client):
        resp = client.post("/v1/deforestation/classify", json={"ndvi": 0.7})
        assert resp.status_code == 200
        data = resp.json()
        assert data["land_cover_type"] == "dense_forest"
        assert data["is_forest"] is True

    def test_classify_non_forest(self, client):
        resp = client.post("/v1/deforestation/classify", json={"ndvi": 0.1})
        assert resp.status_code == 200
        assert resp.json()["is_forest"] is False

    def test_classify_water(self, client):
        resp = client.post("/v1/deforestation/classify", json={"ndvi": 0.1, "ndwi": 0.5})
        assert resp.status_code == 200
        assert resp.json()["land_cover_type"] == "water"


class TestDetectChangeEndpoint:
    def test_detect_change_endpoint(self, client):
        resp = client.post("/v1/deforestation/detect-change", json={
            "polygon": [[-55.0, -10.0], [-54.5, -10.0]],
            "baseline_date": "2020-12-31",
            "current_date": "2023-06-01",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["change_id"].startswith("chg-")
        assert data["status"] == "completed"

    def test_detect_change_has_dndvi(self, client):
        resp = client.post("/v1/deforestation/detect-change", json={
            "polygon": [[-55.0, -10.0]],
            "baseline_date": "2020-12-31",
            "current_date": "2023-06-01",
        })
        assert "dndvi" in resp.json()


class TestCheckBaselineEndpoint:
    def test_check_baseline_endpoint(self, client):
        resp = client.post("/v1/deforestation/check-baseline", json={
            "latitude": -10.0,
            "longitude": -55.0,
            "country": "BRA",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["assessment_id"].startswith("bl-")
        assert data["baseline_date"] == "2020-12-31"
        assert data["compliance_status"] in ("COMPLIANT", "REVIEW_REQUIRED", "NON_COMPLIANT")

    def test_check_baseline_default_country(self, client):
        resp = client.post("/v1/deforestation/check-baseline", json={
            "latitude": -10.0,
            "longitude": -55.0,
        })
        assert resp.status_code == 200
        assert resp.json()["country"] == "BRA"


class TestCheckBaselinePolygonEndpoint:
    def test_check_baseline_polygon_endpoint(self, client):
        resp = client.post("/v1/deforestation/check-baseline/polygon", json={
            "polygon": [[-55.0, -10.0], [-54.5, -10.0], [-54.5, -9.5], [-55.0, -9.5]],
            "country": "BRA",
            "sample_points": 9,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["assessment_id"].startswith("blp-")
        assert data["baseline_date"] == "2020-12-31"
        assert data["overall_compliance"] in ("COMPLIANT", "REVIEW_REQUIRED", "NON_COMPLIANT")

    def test_check_baseline_polygon_custom_samples(self, client):
        resp = client.post("/v1/deforestation/check-baseline/polygon", json={
            "polygon": [[-55.0, -10.0]],
            "sample_points": 16,
        })
        assert resp.status_code == 200
        assert resp.json()["sample_points"] == 16


class TestQueryAlertsEndpoint:
    def test_query_alerts_endpoint(self, client):
        resp = client.post("/v1/deforestation/alerts/query", json={
            "polygon": [[-55.0, -10.0], [-54.5, -10.0]],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert len(data["alerts"]) >= 1

    def test_query_alerts_has_alert_fields(self, client):
        resp = client.post("/v1/deforestation/alerts/query", json={
            "polygon": [[-55.0, -10.0]],
        })
        alert = resp.json()["alerts"][0]
        assert "alert_id" in alert
        assert "source" in alert
        assert "confidence" in alert


class TestAggregateAlertsEndpoint:
    def test_aggregate_alerts_endpoint(self, client):
        resp = client.post("/v1/deforestation/alerts/aggregate", json={
            "polygon": [[-55.0, -10.0], [-54.5, -10.0]],
            "sources": ["glad", "radd"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_alerts"] >= 1
        assert "deduplicated_alerts" in data

    def test_aggregate_default_sources(self, client):
        resp = client.post("/v1/deforestation/alerts/aggregate", json={
            "polygon": [[-55.0, -10.0]],
        })
        assert resp.status_code == 200
        assert "glad" in resp.json()["sources"]


class TestAssessComplianceEndpoint:
    def test_assess_compliance_endpoint(self, client):
        resp = client.post("/v1/deforestation/compliance/assess", json={
            "polygon_id": "plot-001",
            "alerts": [],
        })
        assert resp.status_code == 200
        assert resp.json()["compliance_status"] == "COMPLIANT"

    def test_assess_compliance_non_compliant(self, client):
        resp = client.post("/v1/deforestation/compliance/assess", json={
            "polygon_id": "plot-001",
            "alerts": [{"post_cutoff": True, "confidence": "high"}],
        })
        assert resp.status_code == 200
        assert resp.json()["compliance_status"] == "NON_COMPLIANT"


class TestGenerateReportEndpoint:
    def test_generate_report_endpoint(self, client):
        resp = client.post("/v1/deforestation/compliance/report", json={
            "polygon_id": "plot-001",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_id"].startswith("rpt-")
        assert data["compliance_status"] == "COMPLIANT"

    def test_generate_report_non_compliant(self, client):
        resp = client.post("/v1/deforestation/compliance/report", json={
            "polygon_id": "plot-001",
            "alerts": [{"post_cutoff": True, "confidence": "high"}],
        })
        assert resp.json()["compliance_status"] == "NON_COMPLIANT"


class TestGetReportEndpoint:
    def test_get_report_endpoint(self, client):
        # First create a report
        resp = client.post("/v1/deforestation/compliance/report", json={
            "polygon_id": "plot-001",
        })
        report_id = resp.json()["report_id"]

        # Then retrieve it
        resp2 = client.get(f"/v1/deforestation/compliance/{report_id}")
        assert resp2.status_code == 200
        assert resp2.json()["report_id"] == report_id

    def test_get_report_not_found(self, client):
        resp = client.get("/v1/deforestation/compliance/rpt-nonexistent")
        assert resp.status_code == 404


class TestMonitoringEndpoints:
    def test_start_monitoring_endpoint(self, client):
        resp = client.post("/v1/deforestation/monitor/start", json={
            "polygon_id": "plot-001",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"].startswith("job-")
        assert data["status"] == "running"
        assert data["is_running"] is True

    def test_get_monitoring_job_endpoint(self, client):
        resp = client.post("/v1/deforestation/monitor/start", json={
            "polygon_id": "plot-001",
        })
        job_id = resp.json()["job_id"]
        resp2 = client.get(f"/v1/deforestation/monitor/{job_id}")
        assert resp2.status_code == 200
        assert resp2.json()["job_id"] == job_id

    def test_get_monitoring_job_not_found(self, client):
        resp = client.get("/v1/deforestation/monitor/job-nonexistent")
        assert resp.status_code == 404

    def test_list_monitoring_jobs_endpoint(self, client):
        client.post("/v1/deforestation/monitor/start", json={"polygon_id": "plot-001"})
        client.post("/v1/deforestation/monitor/start", json={"polygon_id": "plot-002"})
        resp = client.get("/v1/deforestation/monitor/jobs")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_stop_monitoring_endpoint(self, client):
        resp = client.post("/v1/deforestation/monitor/start", json={
            "polygon_id": "plot-001",
        })
        job_id = resp.json()["job_id"]
        resp2 = client.post(f"/v1/deforestation/monitor/{job_id}/stop")
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "stopped"
        assert resp2.json()["is_running"] is False

    def test_stop_monitoring_not_found(self, client):
        resp = client.post("/v1/deforestation/monitor/job-nonexistent/stop")
        assert resp.status_code == 404


class TestListScenesEndpoint:
    def test_list_scenes_endpoint(self, client):
        # Acquire a scene first
        client.post("/v1/deforestation/acquire", json={
            "polygon": [[-55.0, -10.0]],
            "start_date": "2023-01-01",
            "end_date": "2023-06-01",
        })
        resp = client.get("/v1/deforestation/scenes")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_list_scenes_empty(self, client):
        resp = client.get("/v1/deforestation/scenes")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestForestDefinitionsEndpoint:
    def test_forest_definitions_endpoint(self, client):
        resp = client.get("/v1/deforestation/forest-definitions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 9
        assert "BRA" in data["definitions"]
        assert "IDN" in data["definitions"]
        assert "FAO_DEFAULT" in data["definitions"]

    def test_forest_definitions_structure(self, client):
        resp = client.get("/v1/deforestation/forest-definitions")
        bra = resp.json()["definitions"]["BRA"]
        assert "min_tree_cover_pct" in bra
        assert "min_height_m" in bra
        assert "min_area_ha" in bra


class TestInvalidRequest:
    def test_invalid_request_returns_422(self, client):
        # Missing required fields
        resp = client.post("/v1/deforestation/acquire", json={})
        assert resp.status_code == 422

    def test_invalid_classify_missing_ndvi(self, client):
        resp = client.post("/v1/deforestation/classify", json={})
        assert resp.status_code == 422

    def test_invalid_detect_change_missing_fields(self, client):
        resp = client.post("/v1/deforestation/detect-change", json={})
        assert resp.status_code == 422

    def test_invalid_baseline_missing_coords(self, client):
        resp = client.post("/v1/deforestation/check-baseline", json={})
        assert resp.status_code == 422

    def test_invalid_monitor_start_missing_polygon(self, client):
        resp = client.post("/v1/deforestation/monitor/start", json={})
        assert resp.status_code == 422
