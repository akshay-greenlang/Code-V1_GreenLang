# -*- coding: utf-8 -*-
"""Load / Performance Tests for Grafana SDK (OBS-002) - ~8 tests.

Validates that dashboard/panel builders meet throughput and latency targets.

Author: GreenLang Platform Team  |  Date: February 2026
"""
import json, time, copy, asyncio, tracemalloc, pytest


# ---------------------------------------------------------------------------
# Inline lightweight stubs (same logic as unit-test stubs, trimmed)
# ---------------------------------------------------------------------------
class _DashboardBuilder:
    GRID_WIDTH = 24
    def __init__(self):
        self._title = "New Dashboard"; self._uid = None; self._tags = []
        self._description = ""; self._time_from = "now-6h"; self._time_to = "now"
        self._refresh = "30s"; self._timezone = "browser"; self._editable = True
        self._schema_version = 39; self._panels = []; self._templating = []
        self._annotations = []; self._next_id = 1; self._cy = 0; self._cx = 0

    def with_title(self, t): self._title = t; return self
    def with_uid(self, u): self._uid = u; return self
    def with_tags(self, t): self._tags = list(t); return self
    def with_description(self, d): self._description = d; return self
    def with_time_range(self, f, t): self._time_from = f; self._time_to = t; return self
    def with_refresh(self, i): self._refresh = i; return self

    def add_panel(self, panel_dict):
        p = copy.deepcopy(panel_dict); p["id"] = self._next_id; self._next_id += 1
        w = p.get("gridPos", {}).get("w", 12); h = p.get("gridPos", {}).get("h", 8)
        if "gridPos" not in p or p["gridPos"].get("x") is None:
            if self._cx + w > self.GRID_WIDTH: self._cx = 0; self._cy += h
            p["gridPos"] = {"x": self._cx, "y": self._cy, "w": w, "h": h}; self._cx += w
        self._panels.append(p); return self

    def add_row(self, title):
        self._panels.append({"id": self._next_id, "type": "row", "title": title,
            "collapsed": False, "gridPos": {"x": 0, "y": self._cy, "w": 24, "h": 1}})
        self._next_id += 1; self._cy += 1; self._cx = 0; return self

    def add_variable(self, v): self._templating.append(v); return self

    def build(self):
        r = {"title": self._title, "tags": self._tags, "editable": self._editable,
            "schemaVersion": self._schema_version,
            "time": {"from": self._time_from, "to": self._time_to},
            "refresh": self._refresh, "timezone": self._timezone,
            "panels": self._panels, "templating": {"list": self._templating},
            "annotations": {"list": self._annotations}}
        if self._uid: r["uid"] = self._uid
        if self._description: r["description"] = self._description
        return r


class _PanelBuilder:
    @classmethod
    def stat(cls): pb = cls(); pb._type = "stat"; return pb
    @classmethod
    def timeseries(cls): pb = cls(); pb._type = "timeseries"; return pb
    @classmethod
    def gauge(cls): pb = cls(); pb._type = "gauge"; return pb
    @classmethod
    def table(cls): pb = cls(); pb._type = "table"; return pb

    def __init__(self):
        self._type = "timeseries"; self._title = ""; self._grid = {"w": 12, "h": 8}
        self._targets = []; self._unit = ""; self._color_mode = "palette-classic"
        self._thresholds = []; self._datasource = None

    def with_title(self, t): self._title = t; return self
    def with_grid_pos(self, w, h, x=None, y=None):
        self._grid = {"w": w, "h": h}
        if x is not None: self._grid["x"] = x
        if y is not None: self._grid["y"] = y
        return self
    def with_unit(self, u): self._unit = u; return self
    def add_target(self, expr, legend="", ref_id=None):
        t = {"expr": expr, "refId": ref_id or chr(65 + len(self._targets))}
        if legend: t["legendFormat"] = legend
        self._targets.append(t); return self
    def with_thresholds(self, steps): self._thresholds = list(steps); return self

    def build(self):
        p = {"type": self._type, "title": self._title, "gridPos": self._grid}
        if self._datasource: p["datasource"] = self._datasource
        if self._targets: p["targets"] = self._targets
        fc = {"color": {"mode": self._color_mode}}
        if self._unit: fc["unit"] = self._unit
        if self._thresholds: fc["thresholds"] = {"mode": "absolute", "steps": self._thresholds}
        p["fieldConfig"] = {"defaults": fc, "overrides": []}
        return p


# ---------------------------------------------------------------------------
# Performance / Load tests
# ---------------------------------------------------------------------------
@pytest.mark.performance
class TestGrafanaLoadPerformance:
    """Load and performance tests for Grafana SDK builders."""

    def test_dashboard_builder_throughput(self):
        """Build 100 dashboards in < 5 seconds."""
        panel_template = _PanelBuilder.stat().with_title("P").with_unit("short").add_target("up").with_grid_pos(6, 4).build()

        start = time.perf_counter()
        for i in range(100):
            db = _DashboardBuilder()
            db.with_title("Dashboard %d" % i).with_uid("perf-%d" % i).with_tags(["perf"])
            db.add_row("Row 1")
            for j in range(10):
                db.add_panel(panel_template)
            d = db.build()
            assert len(d["panels"]) == 11  # 1 row + 10 panels
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, "100 dashboards took %.2fs (target <5s)" % elapsed

    def test_panel_builder_throughput(self):
        """Build 1000 panels in < 2 seconds."""
        start = time.perf_counter()
        for i in range(1000):
            p = (
                _PanelBuilder.timeseries()
                .with_title("Panel %d" % i)
                .with_unit("short")
                .add_target("metric_%d{job='api'}" % i, legend="{{instance}}")
                .with_grid_pos(12, 8)
                .with_thresholds([
                    {"value": None, "color": "green"},
                    {"value": 80, "color": "red"},
                ])
                .build()
            )
            assert p["type"] == "timeseries"
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, "1000 panels took %.2fs (target <2s)" % elapsed

    def test_json_serialization_performance(self):
        """Serialize a large dashboard (50 panels) to JSON 100 times in < 3 seconds."""
        db = _DashboardBuilder()
        db.with_title("Large").with_uid("large-perf")
        for i in range(50):
            db.add_panel(
                _PanelBuilder.timeseries()
                .with_title("Panel %d" % i)
                .add_target("sum(rate(http_requests_total[5m]))")
                .with_grid_pos(12, 8)
                .build()
            )
        dashboard = db.build()

        start = time.perf_counter()
        for _ in range(100):
            text = json.dumps(dashboard, indent=2)
            assert len(text) > 0
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, "100 serializations took %.2fs (target <3s)" % elapsed

    def test_dashboard_validation_throughput(self):
        """Validate 500 dashboards in < 3 seconds (panel type + gridPos checks)."""
        dashboards = []
        for i in range(500):
            db = _DashboardBuilder()
            db.with_title("V%d" % i).with_uid("v-%d" % i)
            for j in range(5):
                db.add_panel({"type": "stat", "title": "P%d" % j, "gridPos": {"w": 6, "h": 4}})
            dashboards.append(db.build())

        valid_types = {"stat", "gauge", "timeseries", "table", "row", "barchart", "piechart", "heatmap", "logs", "text"}
        start = time.perf_counter()
        for d in dashboards:
            for p in d["panels"]:
                assert p.get("type") in valid_types
                gp = p.get("gridPos", {})
                assert 1 <= gp.get("w", 0) <= 24
                assert gp.get("h", 0) >= 1
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, "500 validations took %.2fs (target <3s)" % elapsed

    def test_folder_hierarchy_generation_performance(self):
        """Generate folder hierarchy metadata 10,000 times in < 1 second."""
        hierarchy = [
            {"uid": "gl-00-executive", "title": "00-Executive"},
            {"uid": "gl-01-infrastructure", "title": "01-Infrastructure"},
            {"uid": "gl-02-data-stores", "title": "02-Data-Stores"},
            {"uid": "gl-03-observability", "title": "03-Observability"},
            {"uid": "gl-04-security", "title": "04-Security"},
            {"uid": "gl-05-applications", "title": "05-Applications"},
            {"uid": "gl-06-alerts", "title": "06-Alerts"},
        ]
        start = time.perf_counter()
        for _ in range(10_000):
            uids = {f["uid"] for f in hierarchy}
            assert len(uids) == 7
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, "10K hierarchy iterations took %.2fs (target <1s)" % elapsed

    @pytest.mark.asyncio
    async def test_concurrent_dashboard_builds(self):
        """Build 50 dashboards concurrently via asyncio.gather."""
        async def build_one(idx):
            db = _DashboardBuilder()
            db.with_title("Async-%d" % idx).with_uid("async-%d" % idx)
            for j in range(8):
                db.add_panel(
                    _PanelBuilder.stat().with_title("P%d" % j).with_grid_pos(6, 4).build()
                )
            return db.build()

        start = time.perf_counter()
        results = await asyncio.gather(*[build_one(i) for i in range(50)])
        elapsed = time.perf_counter() - start

        assert len(results) == 50
        assert all(len(r["panels"]) == 8 for r in results)
        assert elapsed < 2.0, "50 concurrent builds took %.2fs (target <2s)" % elapsed

    def test_large_dashboard_build(self):
        """Build a single dashboard with 50 panels and verify structure."""
        db = _DashboardBuilder()
        db.with_title("Mega Dashboard").with_uid("mega-50")
        db.add_row("Section A")
        for i in range(25):
            db.add_panel(
                _PanelBuilder.timeseries()
                .with_title("TS-%d" % i)
                .add_target("rate(http_requests_total[5m])")
                .with_grid_pos(12, 8)
                .build()
            )
        db.add_row("Section B")
        for i in range(25):
            db.add_panel(
                _PanelBuilder.stat()
                .with_title("ST-%d" % i)
                .with_unit("short")
                .with_grid_pos(6, 4)
                .build()
            )

        start = time.perf_counter()
        d = db.build()
        elapsed = time.perf_counter() - start

        assert len(d["panels"]) == 52  # 2 rows + 25 + 25
        assert elapsed < 0.1, "Large dashboard build took %.4fs (target <0.1s)" % elapsed

    def test_memory_usage_dashboard_batch(self):
        """Build 200 dashboards and verify memory increase < 50 MB."""
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        dashboards = []
        for i in range(200):
            db = _DashboardBuilder()
            db.with_title("Mem-%d" % i).with_uid("mem-%d" % i)
            for j in range(10):
                db.add_panel(
                    _PanelBuilder.stat()
                    .with_title("P%d" % j)
                    .add_target("metric_%d" % j)
                    .with_grid_pos(6, 4)
                    .build()
                )
            dashboards.append(db.build())

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate memory difference
        stats_before = {str(s.traceback): s.size for s in snapshot_before.statistics("traceback")}
        total_after = sum(s.size for s in snapshot_after.statistics("traceback"))
        total_before = sum(stats_before.values())
        increase_mb = (total_after - total_before) / (1024 * 1024)

        assert len(dashboards) == 200
        assert increase_mb < 50, "Memory increase %.2f MB exceeds 50 MB limit" % increase_mb
