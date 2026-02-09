# -*- coding: utf-8 -*-
"""
Unit Tests for MonitoringPipelineEngine (AGENT-DATA-007)

Tests the 7-stage pipeline orchestration: INITIALIZATION -> IMAGE_ACQUISITION
-> INDEX_CALCULATION -> CLASSIFICATION -> CHANGE_DETECTION -> ALERT_INTEGRATION
-> REPORT_GENERATION, plus job management, stage tracking, pipeline statistics,
monitoring frequency, and SHA-256 provenance.

Coverage target: 85%+ of monitoring_pipeline.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline MonitoringPipelineEngine mirroring
# greenlang/deforestation_satellite/monitoring_pipeline.py
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    "initialization",
    "image_acquisition",
    "index_calculation",
    "classification",
    "change_detection",
    "alert_integration",
    "report_generation",
]

MONITORING_FREQUENCY_ON_DEMAND = "on_demand"
MONITORING_FREQUENCY_WEEKLY = "weekly"
MONITORING_FREQUENCY_MONTHLY = "monthly"
MONITORING_FREQUENCY_QUARTERLY = "quarterly"

MONITORING_FREQUENCIES = [
    MONITORING_FREQUENCY_ON_DEMAND,
    MONITORING_FREQUENCY_WEEKLY,
    MONITORING_FREQUENCY_MONTHLY,
    MONITORING_FREQUENCY_QUARTERLY,
]


class MonitoringPipelineEngine:
    """7-stage deforestation monitoring pipeline orchestrator."""

    def __init__(self, agent_id: str = "GL-DATA-GEO-003"):
        self._agent_id = agent_id
        self._jobs: Dict[str, Dict[str, Any]] = {}

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # -----------------------------------------------------------------
    # Pipeline orchestration
    # -----------------------------------------------------------------

    def start_monitoring(
        self,
        polygon_id: str,
        frequency: str = MONITORING_FREQUENCY_ON_DEMAND,
        satellite: str = "sentinel2",
        **kwargs,
    ) -> Dict[str, Any]:
        """Create and start a new monitoring job."""
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = {
            "job_id": job_id,
            "polygon_id": polygon_id,
            "frequency": frequency,
            "satellite": satellite,
            "status": "running",
            "is_running": True,
            "current_stage": None,
            "completed_stages": [],
            "stage_results": {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "agent_id": self._agent_id,
            "provenance_hash": None,
        }
        self._jobs[job_id] = job
        return job

    def run_pipeline(self, job_id: str) -> Dict[str, Any]:
        """Execute all 7 stages of the pipeline for a monitoring job."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Unknown job: {job_id}")

        for stage in PIPELINE_STAGES:
            job["current_stage"] = stage
            job["updated_at"] = datetime.utcnow().isoformat()
            result = self._execute_stage(stage, job)
            job["stage_results"][stage] = result
            job["completed_stages"].append(stage)

        job["status"] = "completed"
        job["is_running"] = False
        job["current_stage"] = None
        job["provenance_hash"] = self._hash({
            "job_id": job_id,
            "polygon_id": job["polygon_id"],
            "completed_stages": job["completed_stages"],
        })
        job["updated_at"] = datetime.utcnow().isoformat()
        return job

    def _execute_stage(self, stage: str, job: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single pipeline stage (mock implementation for SDK)."""
        return {
            "stage": stage,
            "status": "completed",
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "result_summary": f"{stage} completed for {job['polygon_id']}",
        }

    # -----------------------------------------------------------------
    # Job management
    # -----------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """Stop a running monitoring job."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Unknown job: {job_id}")
        job["status"] = "stopped"
        job["is_running"] = False
        job["updated_at"] = datetime.utcnow().isoformat()
        return job

    def list_jobs(
        self,
        status: Optional[str] = None,
        polygon_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all monitoring jobs with optional filters."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j["status"] == status]
        if polygon_id:
            jobs = [j for j in jobs if j["polygon_id"] == polygon_id]
        return jobs

    def get_job_count(self) -> int:
        return len(self._jobs)

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        running = sum(1 for j in self._jobs.values() if j["is_running"])
        completed = sum(1 for j in self._jobs.values() if j["status"] == "completed")
        stopped = sum(1 for j in self._jobs.values() if j["status"] == "stopped")
        return {
            "total_jobs": len(self._jobs),
            "running_jobs": running,
            "completed_jobs": completed,
            "stopped_jobs": stopped,
            "agent_id": self._agent_id,
        }

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _hash(self, data: Dict[str, Any]) -> str:
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPipelineStages:
    def test_pipeline_stages_count(self):
        assert len(PIPELINE_STAGES) == 7

    def test_pipeline_stages_order(self):
        expected = [
            "initialization",
            "image_acquisition",
            "index_calculation",
            "classification",
            "change_detection",
            "alert_integration",
            "report_generation",
        ]
        assert PIPELINE_STAGES == expected

    def test_pipeline_stages_first(self):
        assert PIPELINE_STAGES[0] == "initialization"

    def test_pipeline_stages_last(self):
        assert PIPELINE_STAGES[-1] == "report_generation"

    def test_pipeline_stages_unique(self):
        assert len(PIPELINE_STAGES) == len(set(PIPELINE_STAGES))


class TestMonitoringFrequency:
    def test_monitoring_frequency_values(self):
        assert "on_demand" in MONITORING_FREQUENCIES
        assert "weekly" in MONITORING_FREQUENCIES
        assert "monthly" in MONITORING_FREQUENCIES
        assert "quarterly" in MONITORING_FREQUENCIES

    def test_monitoring_frequency_count(self):
        assert len(MONITORING_FREQUENCIES) == 4


class TestStartMonitoring:
    def test_start_monitoring_creates_job(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        assert "job_id" in job
        assert job["job_id"].startswith("job-")

    def test_start_monitoring_sets_polygon(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        assert job["polygon_id"] == "plot-001"

    def test_start_monitoring_default_frequency(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        assert job["frequency"] == MONITORING_FREQUENCY_ON_DEMAND

    def test_start_monitoring_custom_frequency(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001", frequency="weekly")
        assert job["frequency"] == "weekly"

    def test_start_monitoring_default_satellite(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        assert job["satellite"] == "sentinel2"

    def test_start_monitoring_custom_satellite(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001", satellite="landsat8")
        assert job["satellite"] == "landsat8"

    def test_start_monitoring_is_running(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        assert job["is_running"] is True
        assert job["status"] == "running"

    def test_start_monitoring_empty_completed_stages(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        assert job["completed_stages"] == []

    def test_start_monitoring_has_timestamps(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        assert "created_at" in job
        assert "updated_at" in job

    def test_start_monitoring_agent_id(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        assert job["agent_id"] == "GL-DATA-GEO-003"


class TestRunPipeline:
    def test_run_pipeline_all_stages(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        result = engine.run_pipeline(job["job_id"])
        assert len(result["completed_stages"]) == 7
        assert result["completed_stages"] == PIPELINE_STAGES

    def test_run_pipeline_stage_tracking(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        result = engine.run_pipeline(job["job_id"])
        for stage in PIPELINE_STAGES:
            assert stage in result["stage_results"]
            assert result["stage_results"][stage]["status"] == "completed"

    def test_run_pipeline_sets_completed(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        result = engine.run_pipeline(job["job_id"])
        assert result["status"] == "completed"
        assert result["is_running"] is False

    def test_run_pipeline_current_stage_none_after(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        result = engine.run_pipeline(job["job_id"])
        assert result["current_stage"] is None

    def test_run_pipeline_unknown_job_raises(self):
        engine = MonitoringPipelineEngine()
        with pytest.raises(ValueError, match="Unknown job"):
            engine.run_pipeline("job-nonexistent")

    def test_pipeline_result_per_stage(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        result = engine.run_pipeline(job["job_id"])
        for stage in PIPELINE_STAGES:
            stage_result = result["stage_results"][stage]
            assert stage_result["stage"] == stage
            assert "started_at" in stage_result
            assert "completed_at" in stage_result

    def test_pipeline_result_summary_per_stage(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        result = engine.run_pipeline(job["job_id"])
        for stage in PIPELINE_STAGES:
            assert "result_summary" in result["stage_results"][stage]

    def test_pipeline_provenance(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        result = engine.run_pipeline(job["job_id"])
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == 64
        int(result["provenance_hash"], 16)


class TestGetJob:
    def test_get_job_found(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        found = engine.get_job(job["job_id"])
        assert found is not None
        assert found["job_id"] == job["job_id"]

    def test_get_job_not_found(self):
        engine = MonitoringPipelineEngine()
        assert engine.get_job("job-nonexistent") is None


class TestStopJob:
    def test_stop_job(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        stopped = engine.stop_job(job["job_id"])
        assert stopped["status"] == "stopped"
        assert stopped["is_running"] is False

    def test_stop_job_unknown_raises(self):
        engine = MonitoringPipelineEngine()
        with pytest.raises(ValueError, match="Unknown job"):
            engine.stop_job("job-nonexistent")

    def test_stop_job_updates_timestamp(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        original = job["updated_at"]
        stopped = engine.stop_job(job["job_id"])
        # updated_at should be >= original
        assert stopped["updated_at"] >= original


class TestListJobs:
    def test_list_jobs_all(self):
        engine = MonitoringPipelineEngine()
        engine.start_monitoring("plot-001")
        engine.start_monitoring("plot-002")
        jobs = engine.list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_filter_status(self):
        engine = MonitoringPipelineEngine()
        j1 = engine.start_monitoring("plot-001")
        engine.start_monitoring("plot-002")
        engine.stop_job(j1["job_id"])
        running = engine.list_jobs(status="running")
        assert len(running) == 1
        stopped = engine.list_jobs(status="stopped")
        assert len(stopped) == 1

    def test_list_jobs_filter_polygon(self):
        engine = MonitoringPipelineEngine()
        engine.start_monitoring("plot-001")
        engine.start_monitoring("plot-002")
        engine.start_monitoring("plot-001")
        plot1 = engine.list_jobs(polygon_id="plot-001")
        assert len(plot1) == 2

    def test_list_jobs_empty(self):
        engine = MonitoringPipelineEngine()
        assert engine.list_jobs() == []


class TestJobCount:
    def test_job_count_initial(self):
        engine = MonitoringPipelineEngine()
        assert engine.get_job_count() == 0

    def test_job_count_after_creation(self):
        engine = MonitoringPipelineEngine()
        engine.start_monitoring("plot-001")
        assert engine.get_job_count() == 1

    def test_job_count_multiple(self):
        engine = MonitoringPipelineEngine()
        engine.start_monitoring("plot-001")
        engine.start_monitoring("plot-002")
        engine.start_monitoring("plot-003")
        assert engine.get_job_count() == 3


class TestJobIsRunningFlag:
    def test_job_is_running_flag_on_start(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        assert job["is_running"] is True

    def test_job_is_running_flag_after_stop(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        engine.stop_job(job["job_id"])
        updated = engine.get_job(job["job_id"])
        assert updated["is_running"] is False

    def test_job_is_running_flag_after_completion(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        engine.run_pipeline(job["job_id"])
        updated = engine.get_job(job["job_id"])
        assert updated["is_running"] is False


class TestJobCompletion:
    def test_job_completion_status(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        engine.run_pipeline(job["job_id"])
        updated = engine.get_job(job["job_id"])
        assert updated["status"] == "completed"

    def test_job_completion_all_stages_present(self):
        engine = MonitoringPipelineEngine()
        job = engine.start_monitoring("plot-001")
        engine.run_pipeline(job["job_id"])
        updated = engine.get_job(job["job_id"])
        assert len(updated["completed_stages"]) == 7


class TestGetStatistics:
    def test_get_statistics_initial(self):
        engine = MonitoringPipelineEngine()
        stats = engine.get_statistics()
        assert stats["total_jobs"] == 0
        assert stats["running_jobs"] == 0
        assert stats["completed_jobs"] == 0
        assert stats["stopped_jobs"] == 0

    def test_get_statistics_after_operations(self):
        engine = MonitoringPipelineEngine()
        j1 = engine.start_monitoring("plot-001")
        j2 = engine.start_monitoring("plot-002")
        j3 = engine.start_monitoring("plot-003")
        engine.run_pipeline(j1["job_id"])
        engine.stop_job(j2["job_id"])
        stats = engine.get_statistics()
        assert stats["total_jobs"] == 3
        assert stats["running_jobs"] == 1
        assert stats["completed_jobs"] == 1
        assert stats["stopped_jobs"] == 1

    def test_get_statistics_agent_id(self):
        engine = MonitoringPipelineEngine()
        stats = engine.get_statistics()
        assert stats["agent_id"] == "GL-DATA-GEO-003"


class TestCustomAgentId:
    def test_custom_agent_id(self):
        engine = MonitoringPipelineEngine(agent_id="CUSTOM-MON-001")
        assert engine.agent_id == "CUSTOM-MON-001"

    def test_default_agent_id(self):
        engine = MonitoringPipelineEngine()
        assert engine.agent_id == "GL-DATA-GEO-003"

    def test_job_inherits_agent_id(self):
        engine = MonitoringPipelineEngine(agent_id="CUSTOM-MON-001")
        job = engine.start_monitoring("plot-001")
        assert job["agent_id"] == "CUSTOM-MON-001"
