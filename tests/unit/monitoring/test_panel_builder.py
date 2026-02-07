# -*- coding: utf-8 -*-
"""Unit Tests for PanelBuilder (OBS-002) - ~22 tests.

Author: GreenLang Platform Team  |  Date: February 2026
"""
import json, copy, pytest


class _PanelBuilder:
    """Stub fluent panel builder matching PRD section 6.2."""

    # -- factory class methods ------------------------------------------------
    @classmethod
    def stat(cls): return cls()._set_type("stat")
    @classmethod
    def gauge(cls): return cls()._set_type("gauge")
    @classmethod
    def timeseries(cls): return cls()._set_type("timeseries")
    @classmethod
    def table(cls): return cls()._set_type("table")
    @classmethod
    def barchart(cls): return cls()._set_type("barchart")
    @classmethod
    def piechart(cls): return cls()._set_type("piechart")
    @classmethod
    def heatmap(cls): return cls()._set_type("heatmap")
    @classmethod
    def logs(cls): return cls()._set_type("logs")

    VALID_TYPES = frozenset([
        "stat", "gauge", "timeseries", "table",
        "barchart", "piechart", "heatmap", "logs",
        "row", "text", "alertlist", "bargauge", "histogram",
    ])

    def __init__(self):
        self._type = "timeseries"
        self._title = ""
        self._description = ""
        self._grid = {"w": 12, "h": 8}
        self._targets = []
        self._thresholds = []
        self._unit = ""
        self._color_mode = "palette-classic"
        self._overrides = []
        self._mappings = []
        self._legend = {"displayMode": "list", "placement": "bottom"}
        self._tooltip = {"mode": "single"}
        self._axis = {}
        self._decimals = None
        self._min_val = None
        self._max_val = None
        self._no_value = None
        self._links = []
        self._repeat = None
        self._transparent = False
        self._datasource = None

    def _set_type(self, t):
        self._type = t
        return self

    # -- fluent setters -------------------------------------------------------
    def with_title(self, t): self._title = t; return self
    def with_description(self, d): self._description = d; return self
    def with_grid_pos(self, w, h, x=None, y=None):
        self._grid = {"w": w, "h": h}
        if x is not None: self._grid["x"] = x
        if y is not None: self._grid["y"] = y
        return self
    def with_datasource(self, uid, ds_type="prometheus"):
        self._datasource = {"uid": uid, "type": ds_type}
        return self

    def add_target(self, expr, legend="", ref_id=None, datasource=None):
        t = {"expr": expr}
        if legend: t["legendFormat"] = legend
        if ref_id: t["refId"] = ref_id
        else: t["refId"] = chr(65 + len(self._targets))  # A, B, C ...
        if datasource: t["datasource"] = datasource
        self._targets.append(t)
        return self

    def add_loki_target(self, log_query, ref_id=None, datasource_uid=None):
        t = {"expr": log_query, "refId": ref_id or chr(65 + len(self._targets))}
        if datasource_uid:
            t["datasource"] = {"uid": datasource_uid, "type": "loki"}
        self._targets.append(t)
        return self

    def with_thresholds(self, steps):
        self._thresholds = list(steps)
        return self

    def with_unit(self, u): self._unit = u; return self
    def with_color_mode(self, m): self._color_mode = m; return self
    def with_decimals(self, d): self._decimals = d; return self
    def with_min(self, v): self._min_val = v; return self
    def with_max(self, v): self._max_val = v; return self
    def with_no_value(self, v): self._no_value = v; return self

    def add_override(self, matcher, properties):
        self._overrides.append({"matcher": matcher, "properties": properties})
        return self

    def add_mapping(self, mapping):
        self._mappings.append(mapping)
        return self

    def with_legend(self, display_mode="list", placement="bottom", calcs=None):
        self._legend = {"displayMode": display_mode, "placement": placement}
        if calcs: self._legend["calcs"] = list(calcs)
        return self

    def with_tooltip(self, mode="single", sort="none"):
        self._tooltip = {"mode": mode, "sort": sort}
        return self

    def with_axis(self, label=None, placement=None, soft_min=None, soft_max=None):
        a = {}
        if label: a["label"] = label
        if placement: a["placement"] = placement
        if soft_min is not None: a["softMin"] = soft_min
        if soft_max is not None: a["softMax"] = soft_max
        self._axis = a
        return self

    def with_links(self, links): self._links = list(links); return self
    def with_repeat(self, var): self._repeat = var; return self
    def with_transparent(self, v=True): self._transparent = v; return self

    def build(self):
        p = {"type": self._type, "title": self._title, "gridPos": self._grid}
        if self._description: p["description"] = self._description
        if self._datasource: p["datasource"] = self._datasource
        if self._targets: p["targets"] = self._targets
        if self._transparent: p["transparent"] = True
        if self._repeat: p["repeat"] = self._repeat
        if self._links: p["links"] = self._links

        # fieldConfig
        fc_defaults = {}
        if self._unit: fc_defaults["unit"] = self._unit
        if self._decimals is not None: fc_defaults["decimals"] = self._decimals
        if self._min_val is not None: fc_defaults["min"] = self._min_val
        if self._max_val is not None: fc_defaults["max"] = self._max_val
        if self._no_value is not None: fc_defaults["noValue"] = self._no_value
        fc_defaults["color"] = {"mode": self._color_mode}
        if self._thresholds:
            fc_defaults["thresholds"] = {
                "mode": "absolute",
                "steps": self._thresholds,
            }
        if self._mappings:
            fc_defaults["mappings"] = self._mappings

        p["fieldConfig"] = {"defaults": fc_defaults, "overrides": self._overrides}

        # options
        opts = {}
        if self._legend: opts["legend"] = self._legend
        if self._tooltip: opts["tooltip"] = self._tooltip
        if self._axis: opts.update(self._axis)
        if opts: p["options"] = opts

        return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def stat_panel(): return _PanelBuilder.stat()

@pytest.fixture
def ts_panel(): return _PanelBuilder.timeseries()


# ---------------------------------------------------------------------------
# Tests -- Panel type factories
# ---------------------------------------------------------------------------
class TestPanelBuilderTypes:
    """Verify every factory method produces the correct panel type."""

    @pytest.mark.parametrize("factory,expected_type", [
        ("stat", "stat"),
        ("gauge", "gauge"),
        ("timeseries", "timeseries"),
        ("table", "table"),
        ("barchart", "barchart"),
        ("piechart", "piechart"),
        ("heatmap", "heatmap"),
        ("logs", "logs"),
    ])
    def test_factory_panel_type(self, factory, expected_type):
        pb = getattr(_PanelBuilder, factory)()
        assert pb.build()["type"] == expected_type

    def test_valid_types_constant(self):
        """Ensure VALID_TYPES includes all factory types plus extras."""
        factory_types = {"stat", "gauge", "timeseries", "table",
                         "barchart", "piechart", "heatmap", "logs"}
        assert factory_types.issubset(_PanelBuilder.VALID_TYPES)


# ---------------------------------------------------------------------------
# Tests -- Targets (Prometheus / Loki)
# ---------------------------------------------------------------------------
class TestPanelBuilderTargets:
    def test_add_prometheus_target(self, ts_panel):
        p = ts_panel.add_target("up{job='api'}", legend="{{instance}}").build()
        assert len(p["targets"]) == 1
        assert p["targets"][0]["expr"] == "up{job='api'}"
        assert p["targets"][0]["legendFormat"] == "{{instance}}"
        assert p["targets"][0]["refId"] == "A"

    def test_add_multiple_targets_auto_ref_id(self, ts_panel):
        ts_panel.add_target("metric_a").add_target("metric_b").add_target("metric_c")
        p = ts_panel.build()
        assert [t["refId"] for t in p["targets"]] == ["A", "B", "C"]

    def test_add_loki_target(self, ts_panel):
        p = ts_panel.add_loki_target('{job="api"} |= "error"', datasource_uid="loki-main").build()
        assert p["targets"][0]["datasource"]["type"] == "loki"


# ---------------------------------------------------------------------------
# Tests -- Grid position
# ---------------------------------------------------------------------------
class TestPanelBuilderGridPos:
    def test_default_grid_pos(self, stat_panel):
        gp = stat_panel.build()["gridPos"]
        assert gp["w"] == 12 and gp["h"] == 8

    def test_custom_grid_pos(self, stat_panel):
        gp = stat_panel.with_grid_pos(6, 4, x=12, y=0).build()["gridPos"]
        assert gp == {"w": 6, "h": 4, "x": 12, "y": 0}


# ---------------------------------------------------------------------------
# Tests -- Styling (thresholds, units, colors)
# ---------------------------------------------------------------------------
class TestPanelBuilderStyling:
    def test_with_thresholds(self, stat_panel):
        steps = [{"value": None, "color": "green"}, {"value": 80, "color": "orange"}, {"value": 95, "color": "red"}]
        p = stat_panel.with_thresholds(steps).build()
        th = p["fieldConfig"]["defaults"]["thresholds"]
        assert th["mode"] == "absolute"
        assert len(th["steps"]) == 3
        assert th["steps"][2]["color"] == "red"

    def test_with_unit(self, stat_panel):
        p = stat_panel.with_unit("percentunit").build()
        assert p["fieldConfig"]["defaults"]["unit"] == "percentunit"

    def test_with_color_mode(self, ts_panel):
        p = ts_panel.with_color_mode("fixed").build()
        assert p["fieldConfig"]["defaults"]["color"]["mode"] == "fixed"

    def test_with_decimals(self, stat_panel):
        p = stat_panel.with_decimals(2).build()
        assert p["fieldConfig"]["defaults"]["decimals"] == 2

    def test_with_min_max(self, stat_panel):
        p = stat_panel.with_min(0).with_max(100).build()
        d = p["fieldConfig"]["defaults"]
        assert d["min"] == 0 and d["max"] == 100

    def test_with_no_value(self, stat_panel):
        p = stat_panel.with_no_value("N/A").build()
        assert p["fieldConfig"]["defaults"]["noValue"] == "N/A"


# ---------------------------------------------------------------------------
# Tests -- Overrides and Mappings
# ---------------------------------------------------------------------------
class TestPanelBuilderOverrides:
    def test_add_override(self, ts_panel):
        p = ts_panel.add_override(
            {"id": "byName", "options": "errors"},
            [{"id": "color", "value": {"fixedColor": "red", "mode": "fixed"}}]
        ).build()
        assert len(p["fieldConfig"]["overrides"]) == 1
        assert p["fieldConfig"]["overrides"][0]["matcher"]["id"] == "byName"

    def test_add_multiple_overrides(self, ts_panel):
        ts_panel.add_override(
            {"id": "byName", "options": "a"}, [{"id": "color", "value": "red"}]
        ).add_override(
            {"id": "byName", "options": "b"}, [{"id": "color", "value": "blue"}]
        )
        assert len(ts_panel.build()["fieldConfig"]["overrides"]) == 2

    def test_add_mapping(self, stat_panel):
        p = stat_panel.add_mapping({"type": "value", "options": {"0": {"text": "Down", "color": "red"}}}).build()
        assert len(p["fieldConfig"]["defaults"]["mappings"]) == 1


# ---------------------------------------------------------------------------
# Tests -- Options (legend, tooltip, axis)
# ---------------------------------------------------------------------------
class TestPanelBuilderOptions:
    def test_default_legend(self, ts_panel):
        opts = ts_panel.build()["options"]
        assert opts["legend"]["displayMode"] == "list"
        assert opts["legend"]["placement"] == "bottom"

    def test_custom_legend_with_calcs(self, ts_panel):
        p = ts_panel.with_legend("table", "right", calcs=["mean", "max"]).build()
        assert p["options"]["legend"]["calcs"] == ["mean", "max"]
        assert p["options"]["legend"]["placement"] == "right"

    def test_tooltip_mode(self, ts_panel):
        p = ts_panel.with_tooltip("all", sort="desc").build()
        assert p["options"]["tooltip"]["mode"] == "all"
        assert p["options"]["tooltip"]["sort"] == "desc"


# ---------------------------------------------------------------------------
# Tests -- Miscellaneous options
# ---------------------------------------------------------------------------
class TestPanelBuilderMisc:
    def test_with_datasource(self, stat_panel):
        p = stat_panel.with_datasource("thanos-main", "prometheus").build()
        assert p["datasource"]["uid"] == "thanos-main"
        assert p["datasource"]["type"] == "prometheus"

    def test_with_transparent(self, stat_panel):
        p = stat_panel.with_transparent().build()
        assert p["transparent"] is True

    def test_with_repeat_variable(self, stat_panel):
        p = stat_panel.with_repeat("namespace").build()
        assert p["repeat"] == "namespace"

    def test_with_links(self, stat_panel):
        p = stat_panel.with_links([{"title": "Details", "url": "/d/detail?var=${__data.fields.name}"}]).build()
        assert len(p["links"]) == 1

    def test_with_axis(self, ts_panel):
        p = ts_panel.with_axis(label="Requests/s", placement="left", soft_min=0).build()
        assert p["options"]["label"] == "Requests/s"


# ---------------------------------------------------------------------------
# Tests -- Output structure
# ---------------------------------------------------------------------------
class TestPanelBuilderOutput:
    def test_build_output_has_required_keys(self, ts_panel):
        p = ts_panel.with_title("Test").build()
        required = {"type", "title", "gridPos", "fieldConfig"}
        assert required.issubset(p.keys())

    def test_build_serializable_to_json(self, ts_panel):
        p = ts_panel.with_title("J").add_target("up").build()
        roundtrip = json.loads(json.dumps(p))
        assert roundtrip["title"] == "J"
        assert roundtrip["targets"][0]["expr"] == "up"

    def test_chaining_fluent_api(self):
        pb = _PanelBuilder.timeseries()
        result = pb.with_title("T").with_unit("s").with_color_mode("fixed").with_decimals(3)
        assert result is pb
