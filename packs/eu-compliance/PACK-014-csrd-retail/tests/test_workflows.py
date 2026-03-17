# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail Pack - Workflow Tests
=============================================

Tests all 8 retail workflows: store emissions, supply chain assessment,
packaging compliance, product sustainability, food waste tracking,
circular economy, ESRS retail disclosure, and regulatory compliance.

31 tests across 8 test classes.
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
WF_DIR = os.path.join(PACK_ROOT, "workflows")


def _load_module(name: str, subdir: str = "workflows"):
    """Load a module from PACK-014 via importlib to avoid package path issues."""
    path = os.path.join(PACK_ROOT, subdir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load workflow modules
store_emissions_wf = _load_module("store_emissions_workflow")
supply_chain_wf = _load_module("supply_chain_assessment_workflow")
packaging_wf = _load_module("packaging_compliance_workflow")
product_wf = _load_module("product_sustainability_workflow")
food_waste_wf = _load_module("food_waste_tracking_workflow")
circular_wf = _load_module("circular_economy_workflow")
esrs_wf = _load_module("esrs_retail_disclosure_workflow")
regulatory_wf = _load_module("regulatory_compliance_workflow")


def _run_async(coro):
    """Run an async coroutine in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ======================================================================
# 1. StoreEmissionsWorkflow (4 tests)
# ======================================================================


class TestStoreEmissionsWorkflow:
    """Tests for StoreEmissionsWorkflow."""

    def test_init(self):
        wf = store_emissions_wf.StoreEmissionsWorkflow()
        assert wf.workflow_id is not None
        assert len(wf.workflow_id) == 36  # UUID format

    def test_run_basic(self):
        wf = store_emissions_wf.StoreEmissionsWorkflow()
        store = store_emissions_wf.StoreData(
            store_id="S001",
            store_name="Test Store",
            country="DE",
            floor_area_sqm=1000.0,
            employee_count=50,
            energy_records=[
                store_emissions_wf.EnergyRecord(
                    fuel_type="natural_gas", consumption_kwh=50000.0
                )
            ],
            electricity_records=[
                store_emissions_wf.ElectricityRecord(
                    consumption_kwh=200000.0, grid_region="DE"
                )
            ],
        )
        inp = store_emissions_wf.StoreEmissionsInput(stores=[store])
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")
        assert result.workflow_id is not None

    def test_phases_list_has_4(self):
        wf = store_emissions_wf.StoreEmissionsWorkflow()
        store = store_emissions_wf.StoreData(
            store_id="S002",
            store_name="Store 2",
            floor_area_sqm=500.0,
            employee_count=20,
        )
        inp = store_emissions_wf.StoreEmissionsInput(stores=[store])
        result = _run_async(wf.execute(inp))
        assert len(result.phases) == 4

    def test_result_has_provenance(self):
        wf = store_emissions_wf.StoreEmissionsWorkflow()
        store = store_emissions_wf.StoreData(
            store_id="S003",
            store_name="Store 3",
            floor_area_sqm=800.0,
            employee_count=30,
        )
        inp = store_emissions_wf.StoreEmissionsInput(stores=[store])
        result = _run_async(wf.execute(inp))
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64  # SHA-256 hex length


# ======================================================================
# 2. SupplyChainAssessmentWorkflow (4 tests)
# ======================================================================


class TestSupplyChainAssessmentWorkflow:
    """Tests for SupplyChainAssessmentWorkflow."""

    def test_init(self):
        wf = supply_chain_wf.SupplyChainAssessmentWorkflow()
        assert wf.workflow_id is not None

    def test_run(self):
        wf = supply_chain_wf.SupplyChainAssessmentWorkflow()
        supplier = supply_chain_wf.SupplierRecord(name="Test Supplier", country="CN")
        inp = supply_chain_wf.SupplyChainInput(suppliers=[supplier])
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

    def test_5_phases(self):
        wf = supply_chain_wf.SupplyChainAssessmentWorkflow()
        supplier = supply_chain_wf.SupplierRecord(name="Supplier A", country="DE")
        inp = supply_chain_wf.SupplyChainInput(suppliers=[supplier])
        result = _run_async(wf.execute(inp))
        assert len(result.phases) == 5

    def test_provenance(self):
        wf = supply_chain_wf.SupplyChainAssessmentWorkflow()
        supplier = supply_chain_wf.SupplierRecord(name="Supplier B", country="FR")
        inp = supply_chain_wf.SupplyChainInput(suppliers=[supplier])
        result = _run_async(wf.execute(inp))
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ======================================================================
# 3. PackagingComplianceWorkflow (4 tests)
# ======================================================================


class TestPackagingComplianceWorkflow:
    """Tests for PackagingComplianceWorkflow."""

    def test_init(self):
        wf = packaging_wf.PackagingComplianceWorkflow()
        assert wf.workflow_id is not None

    def test_run(self):
        wf = packaging_wf.PackagingComplianceWorkflow()
        item = packaging_wf.PackagingItem(
            material="PET", weight_grams=30.0, annual_units=100000
        )
        inp = packaging_wf.PackagingInput(packaging_items=[item])
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

    def test_4_phases(self):
        wf = packaging_wf.PackagingComplianceWorkflow()
        item = packaging_wf.PackagingItem(
            material="HDPE", weight_grams=20.0, annual_units=50000
        )
        inp = packaging_wf.PackagingInput(packaging_items=[item])
        result = _run_async(wf.execute(inp))
        assert len(result.phases) == 4

    def test_provenance(self):
        wf = packaging_wf.PackagingComplianceWorkflow()
        item = packaging_wf.PackagingItem(
            material="glass", weight_grams=250.0, annual_units=10000
        )
        inp = packaging_wf.PackagingInput(packaging_items=[item])
        result = _run_async(wf.execute(inp))
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ======================================================================
# 4. ProductSustainabilityWorkflow (4 tests)
# ======================================================================


class TestProductSustainabilityWorkflow:
    """Tests for ProductSustainabilityWorkflow."""

    def test_init(self):
        wf = product_wf.ProductSustainabilityWorkflow()
        assert wf.workflow_id is not None

    def test_run(self):
        wf = product_wf.ProductSustainabilityWorkflow()
        prod = product_wf.ProductRecord(name="Test Product")
        inp = product_wf.ProductSustainabilityInput(products=[prod])
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

    def test_4_phases(self):
        wf = product_wf.ProductSustainabilityWorkflow()
        prod = product_wf.ProductRecord(name="Widget")
        inp = product_wf.ProductSustainabilityInput(products=[prod])
        result = _run_async(wf.execute(inp))
        assert len(result.phases) == 4

    def test_provenance(self):
        wf = product_wf.ProductSustainabilityWorkflow()
        prod = product_wf.ProductRecord(name="Gadget")
        inp = product_wf.ProductSustainabilityInput(products=[prod])
        result = _run_async(wf.execute(inp))
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ======================================================================
# 5. FoodWasteTrackingWorkflow (4 tests)
# ======================================================================


class TestFoodWasteTrackingWorkflow:
    """Tests for FoodWasteTrackingWorkflow."""

    def test_init(self):
        wf = food_waste_wf.FoodWasteTrackingWorkflow()
        assert wf.workflow_id is not None

    def test_run(self):
        wf = food_waste_wf.FoodWasteTrackingWorkflow()
        rec = food_waste_wf.WasteRecord(store_id="S001", weight_kg=500.0)
        inp = food_waste_wf.FoodWasteInput(waste_records=[rec])
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

    def test_4_phases(self):
        wf = food_waste_wf.FoodWasteTrackingWorkflow()
        rec = food_waste_wf.WasteRecord(store_id="S002", weight_kg=300.0)
        inp = food_waste_wf.FoodWasteInput(waste_records=[rec])
        result = _run_async(wf.execute(inp))
        assert len(result.phases) == 4

    def test_provenance(self):
        wf = food_waste_wf.FoodWasteTrackingWorkflow()
        rec = food_waste_wf.WasteRecord(store_id="S003", weight_kg=700.0)
        inp = food_waste_wf.FoodWasteInput(waste_records=[rec])
        result = _run_async(wf.execute(inp))
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ======================================================================
# 6. CircularEconomyWorkflow (4 tests)
# ======================================================================


class TestCircularEconomyWorkflow:
    """Tests for CircularEconomyWorkflow."""

    def test_init(self):
        wf = circular_wf.CircularEconomyWorkflow()
        assert wf.workflow_id is not None

    def test_run(self):
        wf = circular_wf.CircularEconomyWorkflow()
        prog = circular_wf.TakeBackProgram(
            name="Bottles", weight_collected_tonnes=10.0
        )
        inp = circular_wf.CircularEconomyInput(take_back_programs=[prog])
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

    def test_4_phases(self):
        wf = circular_wf.CircularEconomyWorkflow()
        prog = circular_wf.TakeBackProgram(
            name="Electronics", weight_collected_tonnes=5.0
        )
        inp = circular_wf.CircularEconomyInput(take_back_programs=[prog])
        result = _run_async(wf.execute(inp))
        assert len(result.phases) == 4

    def test_provenance(self):
        wf = circular_wf.CircularEconomyWorkflow()
        prog = circular_wf.TakeBackProgram(
            name="Textiles", weight_collected_tonnes=2.0
        )
        inp = circular_wf.CircularEconomyInput(take_back_programs=[prog])
        result = _run_async(wf.execute(inp))
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ======================================================================
# 7. ESRSRetailDisclosureWorkflow (4 tests)
# ======================================================================


class TestESRSRetailDisclosureWorkflow:
    """Tests for ESRSRetailDisclosureWorkflow."""

    def test_init(self):
        wf = esrs_wf.ESRSRetailDisclosureWorkflow()
        assert wf.workflow_id is not None

    def test_run(self):
        wf = esrs_wf.ESRSRetailDisclosureWorkflow()
        dp = esrs_wf.ESRSDataPoint(topic="E1", datapoint_id="E1-1")
        inp = esrs_wf.ESRSDisclosureInput(datapoints=[dp])
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")

    def test_4_phases(self):
        wf = esrs_wf.ESRSRetailDisclosureWorkflow()
        dp = esrs_wf.ESRSDataPoint(topic="E5", datapoint_id="E5-1")
        inp = esrs_wf.ESRSDisclosureInput(datapoints=[dp])
        result = _run_async(wf.execute(inp))
        assert len(result.phases) == 4

    def test_provenance(self):
        wf = esrs_wf.ESRSRetailDisclosureWorkflow()
        dp = esrs_wf.ESRSDataPoint(topic="S2", datapoint_id="S2-1")
        inp = esrs_wf.ESRSDisclosureInput(datapoints=[dp])
        result = _run_async(wf.execute(inp))
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ======================================================================
# 8. RegulatoryComplianceWorkflow (3 tests)
# ======================================================================


class TestRegulatoryComplianceWorkflow:
    """Tests for RegulatoryComplianceWorkflow."""

    def test_init(self):
        wf = regulatory_wf.RegulatoryComplianceWorkflow()
        assert wf.workflow_id is not None

    def test_3_phases(self):
        wf = regulatory_wf.RegulatoryComplianceWorkflow()
        company = regulatory_wf.CompanyData(
            name="Retail Corp", employee_count=500
        )
        inp = regulatory_wf.RegulatoryComplianceInput(company_data=company)
        result = _run_async(wf.execute(inp))
        assert len(result.phases) == 3

    def test_run_basic(self):
        wf = regulatory_wf.RegulatoryComplianceWorkflow()
        company = regulatory_wf.CompanyData(
            name="Big Retail",
            employee_count=5000,
            annual_revenue_eur=1_000_000_000.0,
        )
        inp = regulatory_wf.RegulatoryComplianceInput(company_data=company)
        result = _run_async(wf.execute(inp))
        assert result.status.value in ("completed", "partial")
        assert result.provenance_hash != ""
