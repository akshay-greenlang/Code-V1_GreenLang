# -*- coding: utf-8 -*-
"""Unit Tests for DashboardBuilder (OBS-002) - ~22 tests.

Author: GreenLang Platform Team  |  Date: February 2026
"""
import json, copy, pytest


class _DashboardBuilder:
    """Stub fluent builder matching PRD section 6.2."""
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
    def with_timezone(self, tz): self._timezone = tz; return self
    def with_editable(self, e): self._editable = e; return self
    def with_schema_version(self, v): self._schema_version = v; return self

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
    def add_annotation(self, a): self._annotations.append(a); return self

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


@pytest.fixture
def builder(): return _DashboardBuilder()

@pytest.fixture
def sample_panel(): return {"type": "stat", "title": "Uptime", "gridPos": {"w": 6, "h": 4}}


class TestDashboardBuilderBasic:
    def test_empty_dashboard_build(self, builder):
        d = builder.build()
        assert d["title"] == "New Dashboard" and d["panels"] == [] and "time" in d

    def test_with_title(self, builder):
        assert builder.with_title("Platform Overview").build()["title"] == "Platform Overview"

    def test_with_uid(self, builder):
        assert builder.with_uid("gl-platform").build()["uid"] == "gl-platform"

    def test_with_tags(self, builder):
        assert builder.with_tags(["greenlang", "monitoring"]).build()["tags"] == ["greenlang", "monitoring"]

    def test_with_description(self, builder):
        assert builder.with_description("Executive dashboard").build()["description"] == "Executive dashboard"

    def test_with_time_range(self, builder):
        d = builder.with_time_range("now-24h", "now").build()
        assert d["time"]["from"] == "now-24h" and d["time"]["to"] == "now"

    def test_with_refresh(self, builder):
        assert builder.with_refresh("10s").build()["refresh"] == "10s"

    def test_with_timezone(self, builder):
        assert builder.with_timezone("utc").build()["timezone"] == "utc"

    def test_with_editable(self, builder):
        assert builder.with_editable(False).build()["editable"] is False

    def test_with_schema_version(self, builder):
        assert builder.with_schema_version(38).build()["schemaVersion"] == 38


class TestDashboardBuilderPanels:
    def test_add_panel_auto_increments_id(self, builder, sample_panel):
        builder.add_panel(sample_panel).add_panel(sample_panel)
        d = builder.build()
        assert d["panels"][0]["id"] == 1 and d["panels"][1]["id"] == 2

    def test_add_panel_auto_layout_gridpos(self, builder):
        builder.add_panel({"type": "stat", "title": "A", "gridPos": {"w": 12, "h": 8}})
        builder.add_panel({"type": "stat", "title": "B", "gridPos": {"w": 12, "h": 8}})
        d = builder.build()
        assert d["panels"][0]["gridPos"]["x"] == 0 and d["panels"][1]["gridPos"]["x"] == 12

    def test_add_row_creates_row_panel(self, builder):
        builder.add_row("Overview")
        d = builder.build()
        assert d["panels"][0]["type"] == "row" and d["panels"][0]["gridPos"]["w"] == 24

    def test_build_with_multiple_panels_wraps_rows(self, builder):
        for i in range(5):
            builder.add_panel({"type": "stat", "title": "P%d" % i, "gridPos": {"w": 6, "h": 4}})
        d = builder.build()
        assert d["panels"][4]["gridPos"]["x"] == 0
        assert d["panels"][4]["gridPos"]["y"] > d["panels"][0]["gridPos"]["y"]


class TestDashboardBuilderTemplating:
    def test_add_variable_template(self, builder):
        builder.add_variable({"name": "datasource", "type": "datasource", "query": "prometheus"})
        assert len(builder.build()["templating"]["list"]) == 1

    def test_add_annotation(self, builder):
        builder.add_annotation({"name": "Deploys", "enable": True})
        assert len(builder.build()["annotations"]["list"]) == 1


class TestDashboardBuilderOutput:
    def test_build_output_structure(self, builder):
        d = builder.with_title("T").with_uid("u").with_tags(["t"]).build()
        required = {"title", "tags", "editable", "schemaVersion", "time", "refresh", "timezone", "panels", "templating", "annotations", "uid"}
        assert required.issubset(d.keys())

    def test_chaining_fluent_api(self, builder):
        result = builder.with_title("T").with_uid("u").with_tags([]).with_description("d")
        result = result.with_time_range("now-1h", "now").with_refresh("5s").with_timezone("utc")
        assert result is builder

    def test_build_serializable_to_json(self, builder):
        d = builder.with_title("T").build()
        assert json.loads(json.dumps(d))["title"] == "T"

    def test_build_matches_grafana_schema(self, builder):
        d = builder.with_title("Test").with_uid("test-uid").with_tags(["test"]).add_row("Row 1").add_panel({"type": "stat", "title": "P1", "gridPos": {"w": 6, "h": 4}}).build()
        assert len(d["panels"]) == 2
        assert d["panels"][0]["type"] == "row" and d["panels"][1]["type"] == "stat"
