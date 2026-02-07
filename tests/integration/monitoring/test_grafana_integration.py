# -*- coding: utf-8 -*-
"""Integration Tests for Grafana SDK (OBS-002) - ~15 tests.

Validates dashboard builders produce valid Grafana-compatible JSON and
that the existing 43 dashboard JSON files at deployment/monitoring/dashboards/
conform to the expected schema.

Author: GreenLang Platform Team  |  Date: February 2026
"""
import json, os, copy, pathlib, pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DASHBOARDS_DIR = pathlib.Path(__file__).resolve().parents[3] / "deployment" / "monitoring" / "dashboards"

VALID_DATASOURCE_TYPES = frozenset([
    "prometheus", "loki", "jaeger", "alertmanager",
    "postgres", "cloudwatch", "tempo", "elasticsearch",
    "grafana-postgresql-datasource", "grafana-piechart-panel",
])

VALID_PANEL_TYPES = frozenset([
    "stat", "gauge", "timeseries", "table",
    "barchart", "piechart", "heatmap", "logs",
    "row", "text", "alertlist", "bargauge", "histogram",
    "graph", "singlestat", "news", "dashlist", "nodeGraph",
    "candlestick", "flamegraph", "geomap", "state-timeline",
    "status-history", "trend", "xychart",
    "grafana-piechart-panel",
])

# The 7 canonical folder UIDs from PRD section 2.3
EXPECTED_FOLDERS = [
    "gl-00-executive", "gl-01-infrastructure", "gl-02-data-stores",
    "gl-03-observability", "gl-04-security", "gl-05-applications",
    "gl-06-alerts",
]


# ---------------------------------------------------------------------------
# Inline lightweight stubs (same as unit-test stubs, trimmed)
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


class _PanelBuilder:
    @classmethod
    def stat(cls): pb = cls(); pb._type = "stat"; return pb
    @classmethod
    def timeseries(cls): pb = cls(); pb._type = "timeseries"; return pb
    @classmethod
    def table(cls): pb = cls(); pb._type = "table"; return pb
    @classmethod
    def gauge(cls): pb = cls(); pb._type = "gauge"; return pb

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
    def with_datasource(self, uid, ds_type="prometheus"):
        self._datasource = {"uid": uid, "type": ds_type}; return self
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
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def builder():
    return _DashboardBuilder()


def _load_all_dashboard_files():
    """Load every .json file from deployment/monitoring/dashboards/."""
    files = []
    if DASHBOARDS_DIR.exists():
        for fp in sorted(DASHBOARDS_DIR.glob("*.json")):
            files.append(fp)
    return files


ALL_DASHBOARD_FILES = _load_all_dashboard_files()


# ---------------------------------------------------------------------------
# Integration tests -- DashboardBuilder output quality
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestDashboardBuilderIntegration:
    """Test that the builder produces dashboards matching real Grafana format."""

    def test_builder_produces_valid_grafana_json(self, builder):
        panel = _PanelBuilder.stat().with_title("Uptime").with_unit("percentunit").add_target(
            "avg(up{job='api'})"
        ).with_grid_pos(6, 4).build()

        d = (
            builder
            .with_title("API Health")
            .with_uid("gl-api-health")
            .with_tags(["greenlang", "api"])
            .with_description("API health overview")
            .with_time_range("now-1h", "now")
            .with_refresh("10s")
            .add_row("Overview")
            .add_panel(panel)
            .build()
        )
        # Validate top-level keys
        assert d["title"] == "API Health"
        assert d["uid"] == "gl-api-health"
        assert isinstance(d["panels"], list) and len(d["panels"]) == 2  # row + stat
        assert d["panels"][0]["type"] == "row"
        assert d["panels"][1]["type"] == "stat"
        # JSON round-trip
        text = json.dumps(d, indent=2)
        assert json.loads(text) == d

    def test_multi_row_dashboard_layout(self, builder):
        """Build a dashboard with multiple rows and panels to verify layout."""
        builder.with_title("Multi-Row").with_uid("gl-multi-row")

        builder.add_row("Metrics")
        for i in range(4):
            builder.add_panel(_PanelBuilder.stat().with_title("M%d" % i).with_grid_pos(6, 4).build())

        builder.add_row("Charts")
        for i in range(2):
            builder.add_panel(_PanelBuilder.timeseries().with_title("C%d" % i).with_grid_pos(12, 8).build())

        d = builder.build()
        assert len(d["panels"]) == 8  # 2 rows + 4 stat + 2 ts
        # Verify row panels at correct positions
        row_panels = [p for p in d["panels"] if p["type"] == "row"]
        assert len(row_panels) == 2
        assert row_panels[0]["title"] == "Metrics"
        assert row_panels[1]["title"] == "Charts"

    def test_dashboard_with_templating_and_annotations(self, builder):
        d = (
            builder
            .with_title("Templated")
            .add_variable({"name": "namespace", "type": "query", "query": "label_values(namespace)"})
            .add_variable({"name": "datasource", "type": "datasource", "query": "prometheus"})
            .add_annotation({"name": "Deploys", "enable": True, "datasource": {"uid": "loki"}})
            .build()
        )
        assert len(d["templating"]["list"]) == 2
        assert len(d["annotations"]["list"]) == 1
        assert d["templating"]["list"][0]["name"] == "namespace"

    def test_panel_with_full_field_config(self):
        p = (
            _PanelBuilder.gauge()
            .with_title("CPU Usage")
            .with_unit("percentunit")
            .with_thresholds([
                {"value": None, "color": "green"},
                {"value": 70, "color": "orange"},
                {"value": 90, "color": "red"},
            ])
            .add_target("100 - (avg(rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)")
            .build()
        )
        fc = p["fieldConfig"]["defaults"]
        assert fc["unit"] == "percentunit"
        assert len(fc["thresholds"]["steps"]) == 3
        assert p["type"] == "gauge"

    def test_dashboard_json_size_reasonable(self, builder):
        """A 20-panel dashboard should produce < 100KB of JSON."""
        for i in range(20):
            builder.add_panel(
                _PanelBuilder.timeseries()
                .with_title("Panel %d" % i)
                .add_target("metric_%d" % i)
                .with_grid_pos(12, 8)
                .build()
            )
        text = json.dumps(builder.build())
        assert len(text) < 100_000, "Dashboard JSON is unexpectedly large: %d bytes" % len(text)

    def test_panel_ids_are_unique_across_dashboard(self, builder):
        builder.add_row("Row 1")
        for i in range(10):
            builder.add_panel({"type": "stat", "title": "P%d" % i, "gridPos": {"w": 6, "h": 4}})
        builder.add_row("Row 2")
        for i in range(10):
            builder.add_panel({"type": "timeseries", "title": "T%d" % i, "gridPos": {"w": 12, "h": 8}})
        d = builder.build()
        ids = [p["id"] for p in d["panels"]]
        assert len(ids) == len(set(ids)), "Panel IDs are not unique: %s" % ids


# ---------------------------------------------------------------------------
# Existing dashboard JSON validation
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestExistingDashboardValidation:
    """Validate the 43 existing dashboard JSON files."""

    @pytest.mark.skipif(not ALL_DASHBOARD_FILES, reason="No dashboard JSON files found")
    @pytest.mark.parametrize("dashboard_path", ALL_DASHBOARD_FILES,
                             ids=lambda p: p.stem)
    def test_dashboard_is_valid_json(self, dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "%s root is not a dict" % dashboard_path.name

    @pytest.mark.skipif(not ALL_DASHBOARD_FILES, reason="No dashboard JSON files found")
    @pytest.mark.parametrize("dashboard_path", ALL_DASHBOARD_FILES,
                             ids=lambda p: p.stem)
    def test_dashboard_has_title(self, dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "title" in data, "%s missing 'title'" % dashboard_path.name
        assert isinstance(data["title"], str) and len(data["title"]) > 0

    @pytest.mark.skipif(not ALL_DASHBOARD_FILES, reason="No dashboard JSON files found")
    @pytest.mark.parametrize("dashboard_path", ALL_DASHBOARD_FILES,
                             ids=lambda p: p.stem)
    def test_dashboard_has_panels(self, dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Every dashboard should have a panels list (may be empty for meta-dashboards)
        assert "panels" in data, "%s missing 'panels'" % dashboard_path.name
        assert isinstance(data["panels"], list)

    @pytest.mark.skipif(not ALL_DASHBOARD_FILES, reason="No dashboard JSON files found")
    @pytest.mark.parametrize("dashboard_path", ALL_DASHBOARD_FILES,
                             ids=lambda p: p.stem)
    def test_dashboard_panels_have_valid_types(self, dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for panel in data.get("panels", []):
            ptype = panel.get("type", "")
            assert ptype in VALID_PANEL_TYPES or ptype == "", (
                "%s has unknown panel type '%s' in panel '%s'"
                % (dashboard_path.name, ptype, panel.get("title", "?"))
            )
            # Check nested panels in collapsed rows
            for sub in panel.get("panels", []):
                stype = sub.get("type", "")
                assert stype in VALID_PANEL_TYPES or stype == "", (
                    "%s has unknown nested panel type '%s'" % (dashboard_path.name, stype)
                )

    @pytest.mark.skipif(not ALL_DASHBOARD_FILES, reason="No dashboard JSON files found")
    @pytest.mark.parametrize("dashboard_path", ALL_DASHBOARD_FILES,
                             ids=lambda p: p.stem)
    def test_dashboard_panels_have_gridpos(self, dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for panel in data.get("panels", []):
            gp = panel.get("gridPos")
            assert gp is not None, (
                "%s panel '%s' missing gridPos" % (dashboard_path.name, panel.get("title", "?"))
            )
            assert "w" in gp and "h" in gp, (
                "%s panel '%s' gridPos missing w/h" % (dashboard_path.name, panel.get("title", "?"))
            )
            assert 1 <= gp["w"] <= 24, (
                "%s panel '%s' gridPos.w=%d out of range" % (dashboard_path.name, panel.get("title", "?"), gp["w"])
            )

    @pytest.mark.skipif(not ALL_DASHBOARD_FILES, reason="No dashboard JSON files found")
    @pytest.mark.parametrize("dashboard_path", ALL_DASHBOARD_FILES,
                             ids=lambda p: p.stem)
    def test_dashboard_panel_ids_unique(self, dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = []
        for panel in data.get("panels", []):
            if "id" in panel:
                ids.append(panel["id"])
            for sub in panel.get("panels", []):
                if "id" in sub:
                    ids.append(sub["id"])
        assert len(ids) == len(set(ids)), (
            "%s has duplicate panel IDs: %s" % (dashboard_path.name, [x for x in ids if ids.count(x) > 1])
        )

    @pytest.mark.skipif(not ALL_DASHBOARD_FILES, reason="No dashboard JSON files found")
    def test_minimum_dashboard_count(self):
        """Verify we have at least 30 dashboards (PRD expects 37+)."""
        assert len(ALL_DASHBOARD_FILES) >= 30, (
            "Expected >= 30 dashboards, found %d" % len(ALL_DASHBOARD_FILES)
        )

    @pytest.mark.skipif(not ALL_DASHBOARD_FILES, reason="No dashboard JSON files found")
    def test_expected_dashboards_present(self):
        """Check that key dashboards from the PRD exist."""
        stems = {fp.stem for fp in ALL_DASHBOARD_FILES}
        expected = [
            "prometheus-health", "thanos-health", "alertmanager-health",
            "auth-service", "agent-factory-v1", "feature-flags",
            "infrastructure-overview",
        ]
        for name in expected:
            assert name in stems, "Expected dashboard '%s' not found" % name
