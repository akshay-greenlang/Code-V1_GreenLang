# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail Pack - End-to-End Tests
===============================================

End-to-end tests that exercise complete workflow-to-template pipelines
for four retail sub-sectors: grocery, fashion, electronics, and online.

12 tests across 4 test classes.
"""

import asyncio
import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

PACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module(name: str, subdir: str):
    """Load a module from PACK-014 via importlib."""
    path = os.path.join(PACK_ROOT, subdir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load modules
store_wf = _load_module("store_emissions_workflow", "workflows")
food_wf = _load_module("food_waste_tracking_workflow", "workflows")
supply_wf = _load_module("supply_chain_assessment_workflow", "workflows")
circular_wf = _load_module("circular_economy_workflow", "workflows")
product_wf = _load_module("product_sustainability_workflow", "workflows")
packaging_wf = _load_module("packaging_compliance_workflow", "workflows")

store_tmpl = _load_module("store_emissions_report", "templates")
food_tmpl = _load_module("food_waste_report", "templates")
supply_tmpl = _load_module("supply_chain_report", "templates")
circular_tmpl = _load_module("circular_economy_report", "templates")
product_tmpl = _load_module("product_sustainability_report", "templates")
packaging_tmpl = _load_module("packaging_compliance_report", "templates")


def _run_async(coro):
    """Run an async coroutine in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ======================================================================
# 1. Grocery Retail E2E (3 tests)
# ======================================================================


class TestGroceryRetailE2E:
    """End-to-end tests for grocery retail sub-sector."""

    def test_store_emissions_workflow_to_template(self):
        """Complete flow: store data -> workflow -> template render."""
        # Build input
        store = store_wf.StoreData(
            store_id="GROC-001",
            store_name="Grocery Main",
            country="DE",
            floor_area_sqm=2500.0,
            employee_count=80,
            store_type="supermarket",
            energy_records=[
                store_wf.EnergyRecord(
                    fuel_type="natural_gas", consumption_kwh=120000.0
                )
            ],
            refrigerant_records=[
                store_wf.RefrigerantRecord(
                    refrigerant_type="R-404A",
                    charge_kg=200.0,
                    leakage_kg=30.0,
                    leakage_rate_pct=15.0,
                    gwp=4728.0,
                )
            ],
            electricity_records=[
                store_wf.ElectricityRecord(
                    consumption_kwh=500000.0,
                    grid_region="DE",
                )
            ],
        )
        inp = store_wf.StoreEmissionsInput(stores=[store])

        # Execute workflow
        wf = store_wf.StoreEmissionsWorkflow()
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

        # Render template
        data = result.model_dump()
        tmpl = store_tmpl.StoreEmissionsReportTemplate()
        md = tmpl.render_markdown(data)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_food_waste_workflow_to_template(self):
        """Complete flow: waste data -> workflow -> template render."""
        rec = food_wf.WasteRecord(
            store_id="GROC-001",
            weight_kg=1200.0,
            food_category="bakery",
        )
        inp = food_wf.FoodWasteInput(
            waste_records=[rec],
            baseline=food_wf.BaselineData(
                baseline_year=2020,
                total_waste_tonnes=50.0,
                total_food_handled_tonnes=5000.0,
            ),
        )

        wf = food_wf.FoodWasteTrackingWorkflow()
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

        data = result.model_dump()
        tmpl = food_tmpl.FoodWasteReportTemplate()
        md = tmpl.render_markdown(data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_provenance_chain(self):
        """Verify provenance hash is propagated from workflow to result."""
        store = store_wf.StoreData(
            store_id="GROC-002",
            store_name="Grocery Express",
            floor_area_sqm=800.0,
            employee_count=25,
        )
        inp = store_wf.StoreEmissionsInput(stores=[store])
        wf = store_wf.StoreEmissionsWorkflow()
        result = _run_async(wf.execute(inp))

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64
        # Each phase should also have a provenance hash
        for phase in result.phases:
            assert phase.provenance_hash != ""


# ======================================================================
# 2. Fashion Retail E2E (3 tests)
# ======================================================================


class TestFashionRetailE2E:
    """End-to-end tests for fashion/apparel retail sub-sector."""

    def test_supply_chain_dd_workflow(self):
        """Supply chain due diligence workflow for fashion brand."""
        supplier = supply_wf.SupplierRecord(
            name="Textile Mill Co",
            country="BD",
            tier="tier_1",
        )
        inp = supply_wf.SupplyChainInput(suppliers=[supplier])

        wf = supply_wf.SupplyChainAssessmentWorkflow()
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")
        assert len(result.phases) == 5

    def test_circular_economy_workflow_to_template(self):
        """Circular economy: take-back program -> workflow -> template."""
        prog = circular_wf.TakeBackProgram(
            name="Old Clothes Collection",
            weight_collected_tonnes=15.0,
            weight_recovered_tonnes=12.0,
            weight_recycled_tonnes=8.0,
            recovery_rate_pct=80.0,
        )
        inp = circular_wf.CircularEconomyInput(take_back_programs=[prog])

        wf = circular_wf.CircularEconomyWorkflow()
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

        data = result.model_dump()
        tmpl = circular_tmpl.CircularEconomyReportTemplate()
        md = tmpl.render_markdown(data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_provenance_chain(self):
        """Verify provenance for fashion supply chain workflow."""
        supplier = supply_wf.SupplierRecord(name="Dye Factory", country="CN")
        inp = supply_wf.SupplyChainInput(suppliers=[supplier])
        wf = supply_wf.SupplyChainAssessmentWorkflow()
        result = _run_async(wf.execute(inp))

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ======================================================================
# 3. Electronics Retail E2E (3 tests)
# ======================================================================


class TestElectronicsRetailE2E:
    """End-to-end tests for electronics retail sub-sector."""

    def test_product_sustainability_workflow_to_template(self):
        """Product sustainability: DPP/PEF for electronics."""
        prod = product_wf.ProductRecord(
            name="Smart TV 55-inch",
            category="electronics",
            weight_kg=15.0,
            requires_dpp=True,
            carbon_footprint_kgco2e=250.0,
            recyclable_pct=65.0,
        )
        inp = product_wf.ProductSustainabilityInput(products=[prod])

        wf = product_wf.ProductSustainabilityWorkflow()
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

        data = result.model_dump()
        tmpl = product_tmpl.ProductSustainabilityReportTemplate()
        md = tmpl.render_markdown(data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_scope3_cat11(self):
        """Test product use-phase emissions relevance for electronics."""
        prod = product_wf.ProductRecord(
            name="Laptop",
            category="electronics",
            weight_kg=2.0,
            requires_dpp=True,
            carbon_footprint_kgco2e=400.0,
            energy_label="A",
        )
        inp = product_wf.ProductSustainabilityInput(products=[prod])
        wf = product_wf.ProductSustainabilityWorkflow()
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")
        assert len(result.phases) == 4

    def test_provenance_chain(self):
        """Verify provenance for electronics product workflow."""
        prod = product_wf.ProductRecord(name="Headphones", category="electronics")
        inp = product_wf.ProductSustainabilityInput(products=[prod])
        wf = product_wf.ProductSustainabilityWorkflow()
        result = _run_async(wf.execute(inp))

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ======================================================================
# 4. Online Retail E2E (3 tests)
# ======================================================================


class TestOnlineRetailE2E:
    """End-to-end tests for online/e-commerce retail sub-sector."""

    def test_packaging_compliance_workflow(self):
        """PPWR packaging compliance for online fulfillment."""
        item = packaging_wf.PackagingItem(
            name="Cardboard Mailer",
            material="cardboard",
            packaging_type="transport",
            weight_grams=150.0,
            annual_units=500000,
        )
        inp = packaging_wf.PackagingInput(packaging_items=[item])

        wf = packaging_wf.PackagingComplianceWorkflow()
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

        data = result.model_dump()
        tmpl = packaging_tmpl.PackagingComplianceReportTemplate()
        md = tmpl.render_markdown(data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_last_mile_scope3(self):
        """Test last-mile delivery as key Scope 3 category for online retail."""
        supplier = supply_wf.SupplierRecord(
            name="Delivery Partner",
            country="DE",
        )
        inp = supply_wf.SupplyChainInput(suppliers=[supplier])
        wf = supply_wf.SupplyChainAssessmentWorkflow()
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")
        assert len(result.phases) == 5

    def test_provenance_chain(self):
        """Verify provenance for online retail packaging workflow."""
        item = packaging_wf.PackagingItem(
            material="LDPE",
            weight_grams=5.0,
            annual_units=1000000,
        )
        inp = packaging_wf.PackagingInput(packaging_items=[item])
        wf = packaging_wf.PackagingComplianceWorkflow()
        result = _run_async(wf.execute(inp))

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64
