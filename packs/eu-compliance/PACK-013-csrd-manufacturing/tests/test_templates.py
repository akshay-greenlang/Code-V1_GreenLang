# -*- coding: utf-8 -*-
"""
PACK-013 CSRD Manufacturing Pack - Template Tests

Tests the TemplateRegistry and all 8 report templates: initialization,
rendering in markdown/html/json, section content, and provenance hashing.

29 tests across 9 test classes (1 registry + 8 templates).
"""

import importlib.util
import json
import sys
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Dynamic module loading via importlib
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PACK_ROOT / "templates"


def _load_module(module_name: str, file_name: str, search_dir: Path = TEMPLATES_DIR):
    """Load a module dynamically using importlib.util.spec_from_file_location."""
    file_path = search_dir / file_name
    if not file_path.exists():
        pytest.skip(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot create spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Load template modules individually (avoids cross-import issues in __init__)
# ---------------------------------------------------------------------------

process_emissions_mod = _load_module(
    "pack013_tpl_process_emissions",
    "process_emissions_report.py",
)
ProcessEmissionsReportTemplate = process_emissions_mod.ProcessEmissionsReportTemplate

product_pcf_mod = _load_module(
    "pack013_tpl_product_pcf",
    "product_pcf_label.py",
)
ProductPCFLabelTemplate = product_pcf_mod.ProductPCFLabelTemplate

energy_perf_mod = _load_module(
    "pack013_tpl_energy_perf",
    "energy_performance_report.py",
)
EnergyPerformanceReportTemplate = energy_perf_mod.EnergyPerformanceReportTemplate

circular_econ_mod = _load_module(
    "pack013_tpl_circular_econ",
    "circular_economy_report.py",
)
CircularEconomyReportTemplate = circular_econ_mod.CircularEconomyReportTemplate

bat_mod = _load_module(
    "pack013_tpl_bat",
    "bat_compliance_report.py",
)
BATComplianceReportTemplate = bat_mod.BATComplianceReportTemplate

water_mod = _load_module(
    "pack013_tpl_water",
    "water_pollution_report.py",
)
WaterPollutionReportTemplate = water_mod.WaterPollutionReportTemplate

scorecard_mod = _load_module(
    "pack013_tpl_scorecard",
    "manufacturing_scorecard.py",
)
ManufacturingScorecardTemplate = scorecard_mod.ManufacturingScorecardTemplate

decarb_mod = _load_module(
    "pack013_tpl_decarb",
    "decarbonization_roadmap.py",
)
DecarbonizationRoadmapTemplate = decarb_mod.DecarbonizationRoadmapTemplate

# Attempt to load the TemplateRegistry from __init__.py
try:
    templates_init_mod = _load_module(
        "pack013_templates_init",
        "__init__.py",
    )
    TemplateRegistry = templates_init_mod.TemplateRegistry
    TEMPLATE_CATALOG = templates_init_mod.TEMPLATE_CATALOG
    SUPPORTED_FORMATS = templates_init_mod.SUPPORTED_FORMATS
    _REGISTRY_AVAILABLE = True
except Exception:
    _REGISTRY_AVAILABLE = False
    TemplateRegistry = None
    TEMPLATE_CATALOG = None
    SUPPORTED_FORMATS = None


# ---------------------------------------------------------------------------
# Shared sample data used across template tests
# ---------------------------------------------------------------------------

SAMPLE_PROCESS_EMISSIONS_DATA = {
    "company_name": "TestCement GmbH",
    "reporting_year": 2025,
    "scope1_total": 45000.0,
    "process_emissions": 30000.0,
    "combustion_emissions": 15000.0,
    "facilities": [
        {
            "name": "Werk Hamburg",
            "country": "DE",
            "scope1": 25000.0,
            "process": 18000.0,
        },
        {
            "name": "Werk Munich",
            "country": "DE",
            "scope1": 20000.0,
            "process": 12000.0,
        },
    ],
    "process_lines": [
        {
            "line_name": "Kiln A",
            "product": "CEM I",
            "emissions": 15000.0,
            "intensity": 0.85,
        },
    ],
    "cbam_embedded": {
        "products": [
            {
                "product": "CEM I",
                "embedded": 0.85,
                "liability_eur": 42500,
            },
        ],
    },
    "ets_benchmark": {
        "benchmark": 0.766,
        "actual": 0.85,
        "free_allocation": 38000,
        "shortfall": 7000,
    },
    "abatement_measures": [
        {
            "name": "Waste heat recovery",
            "reduction": 2000,
            "status": "IN_PROGRESS",
            "investment": 500000,
        },
    ],
}

SAMPLE_PCF_DATA = {
    "product_id": "PROD-001",
    "product_name": "Steel Beam HEB200",
    "functional_unit": "1 tonne",
    "total_pcf_kgco2e": 1850.0,
    "lifecycle_breakdown": {
        "raw_materials": 800.0,
        "manufacturing": 650.0,
        "distribution": 100.0,
        "use_phase": 50.0,
        "end_of_life": 250.0,
    },
    "bom_hotspots": [
        {
            "component_name": "Iron Ore",
            "material": "Fe2O3",
            "emissions_kgco2e": 500.0,
        },
    ],
    "dpp_data": {
        "pcf_kgco2e": 1850.0,
        "methodology": "ISO 14067:2018",
        "allocation_method": "mass",
        "data_quality_rating": "HIGH",
    },
}

SAMPLE_ENERGY_DATA = {
    "company_name": "TestSteel AG",
    "reporting_year": 2025,
    "total_energy_mwh": 120000.0,
    "energy_mix": {
        "natural_gas": 45000.0,
        "electricity_grid": 55000.0,
        "renewable_ppa": 20000.0,
    },
}

SAMPLE_CIRCULAR_DATA = {
    "company_name": "TestAuto GmbH",
    "reporting_year": 2025,
    "mci_score": 0.42,
    "recycled_content_pct": 35.0,
    "waste_diversion_pct": 78.0,
}

SAMPLE_BAT_DATA = {
    "company_name": "TestChem AG",
    "reporting_year": 2025,
    "facility_name": "Ludwigshafen",
    "applicable_brefs": ["LCP", "CWW"],
    "compliance_status": "PARTIAL",
}

SAMPLE_WATER_DATA = {
    "company_name": "TestPaper GmbH",
    "reporting_year": 2025,
    "total_withdrawal_m3": 5000000.0,
    "total_discharge_m3": 4200000.0,
}

SAMPLE_SCORECARD_DATA = {
    "company_name": "TestMfg Corp",
    "reporting_year": 2025,
    "overall_score": 72.0,
    "kpi_dashboard": [
        {
            "kpi_name": "Emission Intensity",
            "value": "0.85 tCO2e/t",
            "target": "0.70 tCO2e/t",
            "status": "WARNING",
            "trend": "improving",
        },
        {
            "kpi_name": "Energy Intensity",
            "value": "2.1 MWh/t",
            "target": "1.8 MWh/t",
            "status": "WARNING",
            "trend": "stable",
        },
    ],
}

SAMPLE_DECARB_DATA = {
    "company_name": "TestSteel AG",
    "baseline_year": 2023,
    "target_year": 2030,
    "baseline_emissions_tco2e": 100000.0,
    "target_reduction_pct": 42.0,
}


# ===========================================================================
# 1. Template Registry (5 tests)
# ===========================================================================

@pytest.mark.skipif(not _REGISTRY_AVAILABLE, reason="TemplateRegistry not loadable")
class TestTemplateRegistry:
    """Tests for the centralized TemplateRegistry."""

    def test_registry_init(self):
        """Registry initializes with default catalog."""
        registry = TemplateRegistry()
        assert registry.template_count == 8
        assert registry.pack_id == "PACK-013"
        assert registry.pack_name == "CSRD Manufacturing"

    def test_list_templates_returns_all_eight(self):
        """list_templates() returns metadata for all 8 templates."""
        registry = TemplateRegistry()
        templates = registry.list_templates()
        assert len(templates) == 8
        keys = {t["key"] for t in templates}
        expected_keys = {
            "process_emissions_report", "product_pcf_label",
            "energy_performance_report", "circular_economy_report",
            "bat_compliance_report", "water_pollution_report",
            "manufacturing_scorecard", "decarbonization_roadmap",
        }
        assert keys == expected_keys

    def test_get_template_returns_instance(self):
        """get_template() returns a template instance."""
        registry = TemplateRegistry()
        tpl = registry.get_template("process_emissions_report")
        assert tpl is not None
        assert hasattr(tpl, "render_markdown")
        assert hasattr(tpl, "render_html")
        assert hasattr(tpl, "render_json")

    def test_get_template_invalid_key_raises(self):
        """get_template() raises KeyError for unknown key."""
        registry = TemplateRegistry()
        with pytest.raises(KeyError):
            registry.get_template("nonexistent_template")

    def test_filter_by_category(self):
        """list_templates(category=...) filters correctly."""
        registry = TemplateRegistry()
        emissions_templates = registry.list_templates(category="emissions")
        assert len(emissions_templates) >= 1
        for t in emissions_templates:
            assert t["category"] == "emissions"


# ===========================================================================
# 2. Process Emissions Report Template (3 tests)
# ===========================================================================

class TestProcessEmissionsReportTemplate:
    """Tests for ProcessEmissionsReportTemplate."""

    def test_render_markdown_basic(self):
        """render_markdown() returns non-empty markdown string."""
        tpl = ProcessEmissionsReportTemplate()
        md = tpl.render_markdown(SAMPLE_PROCESS_EMISSIONS_DATA)
        assert isinstance(md, str)
        assert len(md) > 100
        assert "Process Emissions Report" in md

    def test_render_has_required_sections(self):
        """Rendered markdown contains key sections."""
        tpl = ProcessEmissionsReportTemplate()
        md = tpl.render_markdown(SAMPLE_PROCESS_EMISSIONS_DATA)
        assert "Scope 1 Summary" in md
        assert "Facility Breakdown" in md
        assert "provenance_hash" in md

    def test_render_html_not_empty(self):
        """render_html() returns a non-empty HTML string."""
        tpl = ProcessEmissionsReportTemplate()
        html = tpl.render_html(SAMPLE_PROCESS_EMISSIONS_DATA)
        assert isinstance(html, str)
        assert "<html" in html
        assert "provenance_hash" in html


# ===========================================================================
# 3. Product PCF Label Template (3 tests)
# ===========================================================================

class TestProductPCFLabelTemplate:
    """Tests for ProductPCFLabelTemplate."""

    def test_render_markdown_basic(self):
        tpl = ProductPCFLabelTemplate()
        md = tpl.render_markdown(SAMPLE_PCF_DATA)
        assert isinstance(md, str)
        assert "Product Carbon Footprint" in md
        assert "1,850.00 kgCO2e" in md

    def test_render_has_required_sections(self):
        tpl = ProductPCFLabelTemplate()
        md = tpl.render_markdown(SAMPLE_PCF_DATA)
        assert "Lifecycle Breakdown" in md
        assert "BOM Hotspots" in md
        assert "Compliance Statement" in md
        assert "provenance_hash" in md

    def test_render_json_not_empty(self):
        tpl = ProductPCFLabelTemplate()
        result = tpl.render_json(SAMPLE_PCF_DATA)
        assert isinstance(result, dict)
        assert "report_type" in result
        assert result["report_type"] == "product_pcf_label"
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 4. Energy Performance Report Template (3 tests)
# ===========================================================================

class TestEnergyPerformanceReportTemplate:
    """Tests for EnergyPerformanceReportTemplate."""

    def test_render_markdown_basic(self):
        tpl = EnergyPerformanceReportTemplate()
        md = tpl.render_markdown(SAMPLE_ENERGY_DATA)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_render_has_sections(self):
        tpl = EnergyPerformanceReportTemplate()
        md = tpl.render_markdown(SAMPLE_ENERGY_DATA)
        # At minimum the title and provenance should exist
        assert "Provenance:" in md or "provenance_hash" in md

    def test_render_html_not_empty(self):
        tpl = EnergyPerformanceReportTemplate()
        html = tpl.render_html(SAMPLE_ENERGY_DATA)
        assert isinstance(html, str)
        assert "<html" in html


# ===========================================================================
# 5. Circular Economy Report Template (3 tests)
# ===========================================================================

class TestCircularEconomyReportTemplate:
    """Tests for CircularEconomyReportTemplate."""

    def test_render_markdown_basic(self):
        tpl = CircularEconomyReportTemplate()
        md = tpl.render_markdown(SAMPLE_CIRCULAR_DATA)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_render_has_sections(self):
        tpl = CircularEconomyReportTemplate()
        md = tpl.render_markdown(SAMPLE_CIRCULAR_DATA)
        assert "Provenance:" in md or "provenance_hash" in md

    def test_render_json_not_empty(self):
        tpl = CircularEconomyReportTemplate()
        result = tpl.render_json(SAMPLE_CIRCULAR_DATA)
        assert isinstance(result, dict)
        assert "provenance_hash" in result


# ===========================================================================
# 6. BAT Compliance Report Template (3 tests)
# ===========================================================================

class TestBATComplianceReportTemplate:
    """Tests for BATComplianceReportTemplate."""

    def test_render_markdown_basic(self):
        tpl = BATComplianceReportTemplate()
        md = tpl.render_markdown(SAMPLE_BAT_DATA)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_render_has_sections(self):
        tpl = BATComplianceReportTemplate()
        md = tpl.render_markdown(SAMPLE_BAT_DATA)
        assert "Provenance:" in md or "provenance_hash" in md

    def test_render_html_not_empty(self):
        tpl = BATComplianceReportTemplate()
        html = tpl.render_html(SAMPLE_BAT_DATA)
        assert isinstance(html, str)
        assert "<html" in html


# ===========================================================================
# 7. Water Pollution Report Template (3 tests)
# ===========================================================================

class TestWaterPollutionReportTemplate:
    """Tests for WaterPollutionReportTemplate."""

    def test_render_markdown_basic(self):
        tpl = WaterPollutionReportTemplate()
        md = tpl.render_markdown(SAMPLE_WATER_DATA)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_render_has_sections(self):
        tpl = WaterPollutionReportTemplate()
        md = tpl.render_markdown(SAMPLE_WATER_DATA)
        assert "Provenance:" in md or "provenance_hash" in md

    def test_render_json_not_empty(self):
        tpl = WaterPollutionReportTemplate()
        result = tpl.render_json(SAMPLE_WATER_DATA)
        assert isinstance(result, dict)
        assert "provenance_hash" in result


# ===========================================================================
# 8. Manufacturing Scorecard Template (3 tests)
# ===========================================================================

class TestManufacturingScorecardTemplate:
    """Tests for ManufacturingScorecardTemplate."""

    def test_render_markdown_basic(self):
        tpl = ManufacturingScorecardTemplate()
        md = tpl.render_markdown(SAMPLE_SCORECARD_DATA)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_render_has_sections(self):
        tpl = ManufacturingScorecardTemplate()
        md = tpl.render_markdown(SAMPLE_SCORECARD_DATA)
        assert "Provenance:" in md or "provenance_hash" in md

    def test_render_html_not_empty(self):
        tpl = ManufacturingScorecardTemplate()
        html = tpl.render_html(SAMPLE_SCORECARD_DATA)
        assert isinstance(html, str)
        assert "<html" in html


# ===========================================================================
# 9. Decarbonization Roadmap Template (3 tests)
# ===========================================================================

class TestDecarbonizationRoadmapTemplate:
    """Tests for DecarbonizationRoadmapTemplate."""

    def test_render_markdown_basic(self):
        tpl = DecarbonizationRoadmapTemplate()
        md = tpl.render_markdown(SAMPLE_DECARB_DATA)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_render_has_sections(self):
        tpl = DecarbonizationRoadmapTemplate()
        md = tpl.render_markdown(SAMPLE_DECARB_DATA)
        assert "Provenance:" in md or "provenance_hash" in md

    def test_render_json_not_empty(self):
        tpl = DecarbonizationRoadmapTemplate()
        result = tpl.render_json(SAMPLE_DECARB_DATA)
        assert isinstance(result, dict)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64
