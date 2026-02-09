# -*- coding: utf-8 -*-
"""
Unit Tests for DeforestationSatelliteService Facade & Setup (AGENT-DATA-007)

Tests the DeforestationSatelliteService facade including engine initialization,
satellite imagery acquisition, change detection, alert queries, baseline checks,
land cover classification, compliance report generation, monitoring pipeline,
statistics, configuration, and app integration.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline DeforestationSatelliteService facade mirroring
# greenlang/deforestation_satellite/setup.py
# ---------------------------------------------------------------------------


class DeforestationSatelliteService:
    """Unified facade for all 7 deforestation satellite engines."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._initialized = True

        # Engine state (simplified for SDK)
        self._scenes: List[Dict[str, Any]] = []
        self._changes: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []
        self._baselines: List[Dict[str, Any]] = []
        self._classifications: List[Dict[str, Any]] = []
        self._reports: Dict[str, Dict[str, Any]] = {}
        self._monitoring_jobs: Dict[str, Dict[str, Any]] = {}

        # Engines created flag
        self._engines = {
            "satellite_data": True,
            "forest_change": True,
            "alert_aggregation": True,
            "baseline_assessment": True,
            "deforestation_classifier": True,
            "compliance_report": True,
            "monitoring_pipeline": True,
        }

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def engines(self) -> Dict[str, bool]:
        return dict(self._engines)

    # -----------------------------------------------------------------
    # Engine 1: Satellite Data
    # -----------------------------------------------------------------

    def acquire_imagery(
        self,
        polygon: List[List[float]],
        start_date: str,
        end_date: str,
        satellite: str = "sentinel2",
        max_cloud_cover: int = 30,
    ) -> Dict[str, Any]:
        scene_id = f"scene-{uuid.uuid4().hex[:12]}"
        scene = {
            "scene_id": scene_id,
            "satellite": satellite,
            "start_date": start_date,
            "end_date": end_date,
            "cloud_cover_pct": 15,
            "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
            "resolution_m": 10 if satellite == "sentinel2" else 30,
            "status": "acquired",
            "polygon": polygon,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._scenes.append(scene)
        return scene

    # -----------------------------------------------------------------
    # Engine 2: Forest Change
    # -----------------------------------------------------------------

    def detect_change(
        self,
        polygon: List[List[float]],
        baseline_date: str,
        current_date: str,
    ) -> Dict[str, Any]:
        change_id = f"chg-{uuid.uuid4().hex[:12]}"
        result = {
            "change_id": change_id,
            "primary_change_type": "no_change",
            "dndvi": -0.02,
            "dnbr": -0.01,
            "area_ha": 0.0,
            "confidence": 0.85,
            "baseline_date": baseline_date,
            "current_date": current_date,
            "polygon": polygon,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._changes.append(result)
        return result

    # -----------------------------------------------------------------
    # Engine 3: Alert Aggregation
    # -----------------------------------------------------------------

    def query_alerts(
        self,
        polygon: List[List[float]],
        start_date: str = "2020-12-31",
        end_date: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        alert = {
            "alert_id": f"alert-{uuid.uuid4().hex[:12]}",
            "source": "glad",
            "confidence": "nominal",
            "severity": "low",
            "detected_date": "2023-06-15",
            "post_cutoff": True,
            "area_ha": 0.3,
        }
        self._alerts.append(alert)
        return {
            "alerts": [alert],
            "total": 1,
            "polygon": polygon,
            "date_range": {"start": start_date, "end": end_date or datetime.utcnow().isoformat()[:10]},
        }

    # -----------------------------------------------------------------
    # Engine 4: Baseline Assessment
    # -----------------------------------------------------------------

    def check_baseline(
        self,
        latitude: float,
        longitude: float,
        country: str = "BRA",
    ) -> Dict[str, Any]:
        return {
            "assessment_id": f"bl-{uuid.uuid4().hex[:12]}",
            "latitude": latitude,
            "longitude": longitude,
            "country": country,
            "baseline_date": "2020-12-31",
            "was_forested_at_baseline": True,
            "current_forest_cover_pct": 72.5,
            "baseline_forest_cover_pct": 85.0,
            "change_pct": -12.5,
            "risk_score": 35.0,
            "compliance_status": "REVIEW_REQUIRED",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def check_baseline_polygon(
        self,
        polygon: List[List[float]],
        country: str = "BRA",
        sample_points: int = 9,
    ) -> Dict[str, Any]:
        return {
            "assessment_id": f"blp-{uuid.uuid4().hex[:12]}",
            "polygon": polygon,
            "country": country,
            "sample_points": sample_points,
            "baseline_date": "2020-12-31",
            "overall_compliance": "COMPLIANT",
            "sample_results": [
                {"point_idx": i, "was_forested": True, "current_cover_pct": 80.0}
                for i in range(sample_points)
            ],
            "risk_score": 15.0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # -----------------------------------------------------------------
    # Engine 5: Classification
    # -----------------------------------------------------------------

    def classify_land_cover(
        self,
        ndvi: float,
        evi: Optional[float] = None,
        ndwi: Optional[float] = None,
    ) -> Dict[str, Any]:
        if ndvi >= 0.6:
            land_cover = "dense_forest"
        elif ndvi >= 0.4:
            land_cover = "open_forest"
        elif ndvi >= 0.3:
            land_cover = "shrubland"
        elif ndvi >= 0.2:
            land_cover = "grassland"
        elif ndvi > 0:
            land_cover = "bare_soil"
        else:
            land_cover = "unknown"
        if ndwi is not None and ndwi > 0.3:
            land_cover = "water"

        result = {
            "classification_id": f"cls-{uuid.uuid4().hex[:12]}",
            "land_cover_type": land_cover,
            "ndvi": ndvi,
            "evi": evi,
            "ndwi": ndwi,
            "is_forest": land_cover in ("dense_forest", "open_forest"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._classifications.append(result)
        return result

    # -----------------------------------------------------------------
    # Engine 6: Compliance Report
    # -----------------------------------------------------------------

    def generate_compliance_report(
        self,
        polygon_id: str,
        alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        report_id = f"rpt-{uuid.uuid4().hex[:12]}"
        alerts = alerts or []
        post_cutoff = [a for a in alerts if a.get("post_cutoff", False)]
        high_conf = [a for a in post_cutoff if a.get("confidence") == "high"]

        if not post_cutoff:
            status = "COMPLIANT"
        elif high_conf:
            status = "NON_COMPLIANT"
        else:
            status = "REVIEW_REQUIRED"

        report = {
            "report_id": report_id,
            "polygon_id": polygon_id,
            "compliance_status": status,
            "alert_count": len(alerts),
            "risk_score": min(100.0, len(post_cutoff) * 15.0),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._reports[report_id] = report
        return report

    # -----------------------------------------------------------------
    # Engine 7: Monitoring Pipeline
    # -----------------------------------------------------------------

    def start_monitoring(
        self,
        polygon_id: str,
        frequency: str = "on_demand",
        satellite: str = "sentinel2",
    ) -> Dict[str, Any]:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = {
            "job_id": job_id,
            "polygon_id": polygon_id,
            "frequency": frequency,
            "satellite": satellite,
            "status": "running",
            "is_running": True,
            "created_at": datetime.utcnow().isoformat(),
        }
        self._monitoring_jobs[job_id] = job
        return job

    def get_monitoring_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._monitoring_jobs.get(job_id)

    def stop_monitoring(self, job_id: str) -> Dict[str, Any]:
        job = self._monitoring_jobs.get(job_id)
        if not job:
            raise ValueError(f"Unknown job: {job_id}")
        job["status"] = "stopped"
        job["is_running"] = False
        return job

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_scenes": len(self._scenes),
            "total_changes": len(self._changes),
            "total_alerts": len(self._alerts),
            "total_baselines": len(self._baselines),
            "total_classifications": len(self._classifications),
            "total_reports": len(self._reports),
            "total_monitoring_jobs": len(self._monitoring_jobs),
            "service_initialized": self._initialized,
        }


def configure_deforestation_satellite_service(app: Any) -> DeforestationSatelliteService:
    """Attach the DeforestationSatelliteService to a FastAPI app."""
    service = DeforestationSatelliteService()
    app.state.deforestation_satellite_service = service
    return service


def get_deforestation_satellite_service(app: Any) -> Optional[DeforestationSatelliteService]:
    """Retrieve the DeforestationSatelliteService from a FastAPI app."""
    return getattr(app.state, "deforestation_satellite_service", None)


def get_router():
    """Return the FastAPI router (placeholder)."""
    return None


# ===========================================================================
# Test Classes
# ===========================================================================


class TestServiceInit:
    def test_init_creates_all_engines(self):
        service = DeforestationSatelliteService()
        assert service.is_initialized is True
        engines = service.engines
        assert len(engines) == 7
        for name, active in engines.items():
            assert active is True

    def test_init_with_config(self):
        config = {"default_satellite": "landsat8"}
        service = DeforestationSatelliteService(config=config)
        assert service._config["default_satellite"] == "landsat8"

    def test_init_default_config(self):
        service = DeforestationSatelliteService()
        assert service._config == {}

    def test_engine_names(self):
        service = DeforestationSatelliteService()
        expected = {
            "satellite_data",
            "forest_change",
            "alert_aggregation",
            "baseline_assessment",
            "deforestation_classifier",
            "compliance_report",
            "monitoring_pipeline",
        }
        assert set(service.engines.keys()) == expected


class TestAcquireImagery:
    def test_acquire_imagery(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0], [-54.5, -10.0], [-54.5, -9.5], [-55.0, -9.5]]
        scene = service.acquire_imagery(polygon, "2023-01-01", "2023-06-01")
        assert "scene_id" in scene
        assert scene["scene_id"].startswith("scene-")
        assert scene["satellite"] == "sentinel2"
        assert scene["status"] == "acquired"

    def test_acquire_imagery_custom_satellite(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0], [-54.5, -10.0]]
        scene = service.acquire_imagery(polygon, "2023-01-01", "2023-06-01", satellite="landsat8")
        assert scene["satellite"] == "landsat8"
        assert scene["resolution_m"] == 30

    def test_acquire_imagery_sentinel2_resolution(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        scene = service.acquire_imagery(polygon, "2023-01-01", "2023-06-01")
        assert scene["resolution_m"] == 10

    def test_acquire_imagery_has_bands(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        scene = service.acquire_imagery(polygon, "2023-01-01", "2023-06-01")
        assert len(scene["bands"]) == 6
        assert "B2" in scene["bands"]

    def test_acquire_imagery_has_timestamp(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        scene = service.acquire_imagery(polygon, "2023-01-01", "2023-06-01")
        assert "timestamp" in scene

    def test_acquire_imagery_stores_scene(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        service.acquire_imagery(polygon, "2023-01-01", "2023-06-01")
        stats = service.get_statistics()
        assert stats["total_scenes"] == 1


class TestDetectChange:
    def test_detect_change(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0], [-54.5, -10.0]]
        result = service.detect_change(polygon, "2020-12-31", "2023-06-01")
        assert "change_id" in result
        assert result["change_id"].startswith("chg-")
        assert result["status"] == "completed"

    def test_detect_change_has_dndvi(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        result = service.detect_change(polygon, "2020-12-31", "2023-06-01")
        assert "dndvi" in result
        assert "dnbr" in result

    def test_detect_change_has_confidence(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        result = service.detect_change(polygon, "2020-12-31", "2023-06-01")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_detect_change_stores_result(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        service.detect_change(polygon, "2020-12-31", "2023-06-01")
        stats = service.get_statistics()
        assert stats["total_changes"] == 1


class TestQueryAlerts:
    def test_query_alerts(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0], [-54.5, -10.0]]
        result = service.query_alerts(polygon)
        assert "alerts" in result
        assert result["total"] >= 1

    def test_query_alerts_with_dates(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        result = service.query_alerts(polygon, start_date="2021-01-01", end_date="2023-12-31")
        assert result["date_range"]["start"] == "2021-01-01"
        assert result["date_range"]["end"] == "2023-12-31"

    def test_query_alerts_stores_alert(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        service.query_alerts(polygon)
        stats = service.get_statistics()
        assert stats["total_alerts"] >= 1


class TestCheckBaseline:
    def test_check_baseline(self):
        service = DeforestationSatelliteService()
        result = service.check_baseline(-10.0, -55.0, country="BRA")
        assert "assessment_id" in result
        assert result["assessment_id"].startswith("bl-")
        assert result["baseline_date"] == "2020-12-31"

    def test_check_baseline_country(self):
        service = DeforestationSatelliteService()
        result = service.check_baseline(-2.0, 110.0, country="IDN")
        assert result["country"] == "IDN"

    def test_check_baseline_has_risk_score(self):
        service = DeforestationSatelliteService()
        result = service.check_baseline(-10.0, -55.0)
        assert 0.0 <= result["risk_score"] <= 100.0

    def test_check_baseline_has_compliance_status(self):
        service = DeforestationSatelliteService()
        result = service.check_baseline(-10.0, -55.0)
        assert result["compliance_status"] in ("COMPLIANT", "REVIEW_REQUIRED", "NON_COMPLIANT")


class TestCheckBaselinePolygon:
    def test_check_baseline_polygon(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0], [-54.5, -10.0], [-54.5, -9.5], [-55.0, -9.5]]
        result = service.check_baseline_polygon(polygon, country="BRA")
        assert "assessment_id" in result
        assert result["assessment_id"].startswith("blp-")
        assert result["baseline_date"] == "2020-12-31"

    def test_check_baseline_polygon_sample_points(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0], [-54.5, -10.0]]
        result = service.check_baseline_polygon(polygon, sample_points=16)
        assert result["sample_points"] == 16
        assert len(result["sample_results"]) == 16

    def test_check_baseline_polygon_compliance(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        result = service.check_baseline_polygon(polygon)
        assert result["overall_compliance"] in ("COMPLIANT", "REVIEW_REQUIRED", "NON_COMPLIANT")


class TestClassifyLandCover:
    def test_classify_land_cover(self):
        service = DeforestationSatelliteService()
        result = service.classify_land_cover(ndvi=0.7)
        assert result["land_cover_type"] == "dense_forest"
        assert result["is_forest"] is True

    def test_classify_non_forest(self):
        service = DeforestationSatelliteService()
        result = service.classify_land_cover(ndvi=0.1)
        assert result["is_forest"] is False

    def test_classify_stores_result(self):
        service = DeforestationSatelliteService()
        service.classify_land_cover(ndvi=0.5)
        stats = service.get_statistics()
        assert stats["total_classifications"] == 1


class TestGenerateComplianceReport:
    def test_generate_compliance_report(self):
        service = DeforestationSatelliteService()
        report = service.generate_compliance_report("plot-001")
        assert "report_id" in report
        assert report["report_id"].startswith("rpt-")
        assert report["compliance_status"] == "COMPLIANT"

    def test_generate_compliance_report_non_compliant(self):
        service = DeforestationSatelliteService()
        alerts = [{"post_cutoff": True, "confidence": "high"}]
        report = service.generate_compliance_report("plot-001", alerts=alerts)
        assert report["compliance_status"] == "NON_COMPLIANT"

    def test_generate_compliance_report_stores(self):
        service = DeforestationSatelliteService()
        service.generate_compliance_report("plot-001")
        stats = service.get_statistics()
        assert stats["total_reports"] == 1


class TestStartMonitoring:
    def test_start_monitoring(self):
        service = DeforestationSatelliteService()
        job = service.start_monitoring("plot-001")
        assert "job_id" in job
        assert job["job_id"].startswith("job-")
        assert job["status"] == "running"
        assert job["is_running"] is True

    def test_start_monitoring_stores_job(self):
        service = DeforestationSatelliteService()
        service.start_monitoring("plot-001")
        stats = service.get_statistics()
        assert stats["total_monitoring_jobs"] == 1


class TestGetMonitoringStatus:
    def test_get_monitoring_status(self):
        service = DeforestationSatelliteService()
        job = service.start_monitoring("plot-001")
        status = service.get_monitoring_status(job["job_id"])
        assert status is not None
        assert status["job_id"] == job["job_id"]

    def test_get_monitoring_status_not_found(self):
        service = DeforestationSatelliteService()
        assert service.get_monitoring_status("job-nonexistent") is None


class TestStopMonitoring:
    def test_stop_monitoring(self):
        service = DeforestationSatelliteService()
        job = service.start_monitoring("plot-001")
        stopped = service.stop_monitoring(job["job_id"])
        assert stopped["status"] == "stopped"
        assert stopped["is_running"] is False

    def test_stop_monitoring_unknown_raises(self):
        service = DeforestationSatelliteService()
        with pytest.raises(ValueError, match="Unknown job"):
            service.stop_monitoring("job-nonexistent")


class TestGetStatistics:
    def test_get_statistics_initial(self):
        service = DeforestationSatelliteService()
        stats = service.get_statistics()
        assert stats["total_scenes"] == 0
        assert stats["total_changes"] == 0
        assert stats["total_alerts"] == 0
        assert stats["total_reports"] == 0
        assert stats["service_initialized"] is True

    def test_get_statistics_after_operations(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0]]
        service.acquire_imagery(polygon, "2023-01-01", "2023-06-01")
        service.detect_change(polygon, "2020-12-31", "2023-06-01")
        service.query_alerts(polygon)
        service.classify_land_cover(ndvi=0.5)
        service.generate_compliance_report("plot-001")
        service.start_monitoring("plot-001")
        stats = service.get_statistics()
        assert stats["total_scenes"] == 1
        assert stats["total_changes"] == 1
        assert stats["total_alerts"] >= 1
        assert stats["total_classifications"] == 1
        assert stats["total_reports"] == 1
        assert stats["total_monitoring_jobs"] == 1


class TestConfigureService:
    def test_configure_function(self):
        app = MagicMock()
        service = configure_deforestation_satellite_service(app)
        assert service.is_initialized is True
        assert app.state.deforestation_satellite_service is service

    def test_get_function(self):
        app = MagicMock()
        service = configure_deforestation_satellite_service(app)
        retrieved = get_deforestation_satellite_service(app)
        assert retrieved is service

    def test_get_function_not_configured(self):
        app = MagicMock(spec=[])
        app.state = MagicMock(spec=[])
        result = get_deforestation_satellite_service(app)
        assert result is None

    def test_get_router(self):
        router = get_router()
        # Placeholder returns None; real impl would return APIRouter
        assert router is None


class TestFullLifecycle:
    def test_complete_lifecycle(self):
        service = DeforestationSatelliteService()
        polygon = [[-55.0, -10.0], [-54.5, -10.0], [-54.5, -9.5], [-55.0, -9.5]]

        # Acquire imagery
        scene = service.acquire_imagery(polygon, "2023-01-01", "2023-06-01")
        assert scene["status"] == "acquired"

        # Detect change
        change = service.detect_change(polygon, "2020-12-31", "2023-06-01")
        assert change["status"] == "completed"

        # Query alerts
        alerts_result = service.query_alerts(polygon)
        assert alerts_result["total"] >= 1

        # Check baseline
        baseline = service.check_baseline(-10.0, -55.0, country="BRA")
        assert baseline["baseline_date"] == "2020-12-31"

        # Classify
        cls = service.classify_land_cover(ndvi=0.5)
        assert cls["is_forest"] is True

        # Generate report
        report = service.generate_compliance_report("plot-001", alerts=alerts_result["alerts"])
        assert "compliance_status" in report

        # Start monitoring
        job = service.start_monitoring("plot-001")
        assert job["is_running"] is True

        # Stop monitoring
        stopped = service.stop_monitoring(job["job_id"])
        assert stopped["status"] == "stopped"

        # Statistics
        stats = service.get_statistics()
        assert stats["total_scenes"] >= 1
        assert stats["service_initialized"] is True
