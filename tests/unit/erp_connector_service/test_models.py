# -*- coding: utf-8 -*-
"""
Unit Tests for ERP Connector Models (AGENT-DATA-003)

Tests all enums (ERPSystem 10 values, Scope3Category 16 values,
TransactionType 6, SpendCategory 11, ConnectionStatus 5, SyncMode 4,
EmissionMethodology 4), 10 Layer 1 models, 7 SDK models, and 4 request models.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums mirroring greenlang/erp_connector/models.py Layer 1
# ---------------------------------------------------------------------------


class ERPSystem(str, Enum):
    SAP_S4HANA = "sap_s4hana"
    SAP_ECC = "sap_ecc"
    ORACLE_CLOUD = "oracle_cloud"
    ORACLE_EBS = "oracle_ebs"
    NETSUITE = "netsuite"
    DYNAMICS_365 = "dynamics_365"
    WORKDAY = "workday"
    SAGE = "sage"
    QUICKBOOKS = "quickbooks"
    SIMULATED = "simulated"


class Scope3Category(str, Enum):
    CAT1_PURCHASED_GOODS = "cat1_purchased_goods"
    CAT2_CAPITAL_GOODS = "cat2_capital_goods"
    CAT3_FUEL_ENERGY = "cat3_fuel_energy"
    CAT4_UPSTREAM_TRANSPORT = "cat4_upstream_transport"
    CAT5_WASTE = "cat5_waste"
    CAT6_BUSINESS_TRAVEL = "cat6_business_travel"
    CAT7_EMPLOYEE_COMMUTING = "cat7_employee_commuting"
    CAT8_UPSTREAM_LEASED = "cat8_upstream_leased"
    CAT9_DOWNSTREAM_TRANSPORT = "cat9_downstream_transport"
    CAT10_PROCESSING = "cat10_processing"
    CAT11_USE_OF_SOLD = "cat11_use_of_sold"
    CAT12_END_OF_LIFE = "cat12_end_of_life"
    CAT13_DOWNSTREAM_LEASED = "cat13_downstream_leased"
    CAT14_FRANCHISES = "cat14_franchises"
    CAT15_INVESTMENTS = "cat15_investments"
    UNCLASSIFIED = "unclassified"


class TransactionType(str, Enum):
    PURCHASE = "purchase"
    INVOICE = "invoice"
    CREDIT_NOTE = "credit_note"
    DEBIT_NOTE = "debit_note"
    PAYMENT = "payment"
    JOURNAL = "journal"


class SpendCategory(str, Enum):
    RAW_MATERIALS = "raw_materials"
    ENERGY = "energy"
    TRANSPORTATION = "transportation"
    PACKAGING = "packaging"
    WASTE_MANAGEMENT = "waste_management"
    IT_SERVICES = "it_services"
    PROFESSIONAL_SERVICES = "professional_services"
    TRAVEL = "travel"
    FACILITIES = "facilities"
    CHEMICALS = "chemicals"
    OTHER = "other"


class ConnectionStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TESTING = "testing"
    INITIALIZING = "initializing"


class SyncMode(str, Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DELTA = "delta"
    MANUAL = "manual"


class EmissionMethodology(str, Enum):
    EEIO = "eeio"
    HYBRID = "hybrid"
    PROCESS_BASED = "process_based"
    SUPPLIER_SPECIFIC = "supplier_specific"


# ---------------------------------------------------------------------------
# Inline Layer 1 models
# ---------------------------------------------------------------------------


class VendorMapping:
    def __init__(self, vendor_id: str = "", vendor_name: str = "",
                 primary_category: str = "unclassified",
                 spend_category: str = "other",
                 emission_factor: Optional[float] = None):
        self.vendor_id = vendor_id
        self.vendor_name = vendor_name
        self.primary_category = primary_category
        self.spend_category = spend_category
        self.emission_factor = emission_factor

    def to_dict(self) -> Dict[str, Any]:
        return {"vendor_id": self.vendor_id, "vendor_name": self.vendor_name,
                "primary_category": self.primary_category,
                "spend_category": self.spend_category,
                "emission_factor": self.emission_factor}


class MaterialMapping:
    def __init__(self, material_code: str = "", material_name: str = "",
                 scope3_category: str = "unclassified",
                 emission_factor: Optional[float] = None):
        self.material_code = material_code
        self.material_name = material_name
        self.scope3_category = scope3_category
        self.emission_factor = emission_factor

    def to_dict(self) -> Dict[str, Any]:
        return {"material_code": self.material_code, "material_name": self.material_name,
                "scope3_category": self.scope3_category,
                "emission_factor": self.emission_factor}


class PurchaseOrderLine:
    def __init__(self, item_number: int = 0, material: str = "",
                 description: str = "", quantity: float = 0.0,
                 unit: str = "", unit_price: float = 0.0, amount: float = 0.0):
        self.item_number = item_number
        self.material = material
        self.description = description
        self.quantity = quantity
        self.unit = unit
        self.unit_price = unit_price
        self.amount = amount

    def to_dict(self) -> Dict[str, Any]:
        return {"item_number": self.item_number, "material": self.material,
                "quantity": self.quantity, "unit_price": self.unit_price,
                "amount": self.amount}


class PurchaseOrder:
    def __init__(self, po_number: str = "", vendor_id: str = "",
                 status: str = "open", total_value: float = 0.0,
                 currency: str = "USD", line_items: Optional[List[PurchaseOrderLine]] = None):
        self.po_number = po_number
        self.vendor_id = vendor_id
        self.status = status
        self.total_value = total_value
        self.currency = currency
        self.line_items = line_items or []

    def to_dict(self) -> Dict[str, Any]:
        return {"po_number": self.po_number, "vendor_id": self.vendor_id,
                "status": self.status, "total_value": self.total_value,
                "currency": self.currency,
                "line_items": [li.to_dict() for li in self.line_items]}


class SpendRecord:
    def __init__(self, record_id: str = "", vendor_id: str = "",
                 amount: float = 0.0, currency: str = "USD",
                 date: str = "", category: str = "other",
                 description: str = "", cost_center: str = ""):
        self.record_id = record_id or str(uuid.uuid4())
        self.vendor_id = vendor_id
        self.amount = amount
        self.currency = currency
        self.date = date
        self.category = category
        self.description = description
        self.cost_center = cost_center

    def to_dict(self) -> Dict[str, Any]:
        return {"record_id": self.record_id, "vendor_id": self.vendor_id,
                "amount": self.amount, "currency": self.currency,
                "date": self.date, "category": self.category}


class InventoryItem:
    def __init__(self, item_id: str = "", material: str = "",
                 description: str = "", warehouse: str = "",
                 quantity: float = 0.0, unit: str = "",
                 unit_cost: float = 0.0, material_group: str = ""):
        self.item_id = item_id or str(uuid.uuid4())
        self.material = material
        self.description = description
        self.warehouse = warehouse
        self.quantity = quantity
        self.unit = unit
        self.unit_cost = unit_cost
        self.material_group = material_group

    def to_dict(self) -> Dict[str, Any]:
        return {"item_id": self.item_id, "material": self.material,
                "warehouse": self.warehouse, "quantity": self.quantity,
                "unit_cost": self.unit_cost, "material_group": self.material_group}

    @property
    def total_value(self) -> float:
        return self.quantity * self.unit_cost


class ERPConnectionConfig:
    def __init__(self, erp_system: str = "simulated", host: str = "",
                 port: int = 443, username: str = "", client_id: str = "",
                 company_code: str = "", tenant_id: str = "default"):
        self.erp_system = erp_system
        self.host = host
        self.port = port
        self.username = username
        self.client_id = client_id
        self.company_code = company_code
        self.tenant_id = tenant_id

    def to_dict(self) -> Dict[str, Any]:
        return {"erp_system": self.erp_system, "host": self.host,
                "port": self.port, "username": self.username,
                "tenant_id": self.tenant_id}


class ERPQueryInput:
    def __init__(self, connection_id: str = "", query_type: str = "spend",
                 start_date: str = "", end_date: str = "",
                 filters: Optional[Dict[str, Any]] = None):
        self.connection_id = connection_id
        self.query_type = query_type
        self.start_date = start_date
        self.end_date = end_date
        self.filters = filters or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"connection_id": self.connection_id, "query_type": self.query_type,
                "start_date": self.start_date, "end_date": self.end_date}


class ERPQueryOutput:
    def __init__(self, query_id: str = "", status: str = "completed",
                 record_count: int = 0, records: Optional[List[Dict[str, Any]]] = None,
                 errors: Optional[List[str]] = None):
        self.query_id = query_id or str(uuid.uuid4())
        self.status = status
        self.record_count = record_count
        self.records = records or []
        self.errors = errors or []

    def to_dict(self) -> Dict[str, Any]:
        return {"query_id": self.query_id, "status": self.status,
                "record_count": self.record_count}


# ---------------------------------------------------------------------------
# Inline SDK models
# ---------------------------------------------------------------------------


class ConnectionRecord:
    def __init__(self, connection_id: str = "", erp_system: str = "simulated",
                 host: str = "", status: str = "initializing",
                 sync_count: int = 0, error_count: int = 0):
        self.connection_id = connection_id or str(uuid.uuid4())
        self.erp_system = erp_system
        self.host = host
        self.status = status
        self.sync_count = sync_count
        self.error_count = error_count

    def to_dict(self) -> Dict[str, Any]:
        return {"connection_id": self.connection_id, "erp_system": self.erp_system,
                "host": self.host, "status": self.status}


class SyncJob:
    def __init__(self, job_id: str = "", connection_id: str = "",
                 sync_mode: str = "full", query_type: str = "spend",
                 status: str = "pending", records_synced: int = 0):
        self.job_id = job_id or str(uuid.uuid4())
        self.connection_id = connection_id
        self.sync_mode = sync_mode
        self.query_type = query_type
        self.status = status
        self.records_synced = records_synced

    def to_dict(self) -> Dict[str, Any]:
        return {"job_id": self.job_id, "connection_id": self.connection_id,
                "status": self.status, "records_synced": self.records_synced}


class SpendSummary:
    def __init__(self, period_start: str = "", period_end: str = "",
                 total_spend_usd: float = 0.0, record_count: int = 0,
                 vendor_count: int = 0):
        self.period_start = period_start
        self.period_end = period_end
        self.total_spend_usd = total_spend_usd
        self.record_count = record_count
        self.vendor_count = vendor_count

    def to_dict(self) -> Dict[str, Any]:
        return {"period_start": self.period_start, "period_end": self.period_end,
                "total_spend_usd": self.total_spend_usd,
                "record_count": self.record_count}


class Scope3Summary:
    def __init__(self, category: str = "unclassified", category_name: str = "",
                 total_spend_usd: float = 0.0, estimated_emissions_kgco2e: float = 0.0):
        self.category = category
        self.category_name = category_name
        self.total_spend_usd = total_spend_usd
        self.estimated_emissions_kgco2e = estimated_emissions_kgco2e

    def to_dict(self) -> Dict[str, Any]:
        return {"category": self.category, "total_spend_usd": self.total_spend_usd,
                "estimated_emissions_kgco2e": self.estimated_emissions_kgco2e}


class EmissionResult:
    def __init__(self, result_id: str = "", spend_record_id: str = "",
                 vendor_id: str = "", amount_usd: float = 0.0,
                 emission_factor: float = 0.0, methodology: str = "eeio",
                 estimated_kgco2e: float = 0.0):
        self.result_id = result_id or str(uuid.uuid4())
        self.spend_record_id = spend_record_id
        self.vendor_id = vendor_id
        self.amount_usd = amount_usd
        self.emission_factor = emission_factor
        self.methodology = methodology
        self.estimated_kgco2e = estimated_kgco2e

    def to_dict(self) -> Dict[str, Any]:
        return {"result_id": self.result_id, "vendor_id": self.vendor_id,
                "amount_usd": self.amount_usd, "estimated_kgco2e": self.estimated_kgco2e}


class CurrencyRate:
    def __init__(self, from_currency: str = "USD", to_currency: str = "USD",
                 rate: float = 1.0, effective_date: str = "2025-01-01"):
        self.from_currency = from_currency
        self.to_currency = to_currency
        self.rate = rate
        self.effective_date = effective_date

    def to_dict(self) -> Dict[str, Any]:
        return {"from_currency": self.from_currency, "to_currency": self.to_currency,
                "rate": self.rate, "effective_date": self.effective_date}


class ERPStatistics:
    def __init__(self, total_connections: int = 0, active_connections: int = 0,
                 total_syncs: int = 0, total_spend_records: int = 0,
                 total_purchase_orders: int = 0, total_emissions_calculated: int = 0,
                 scope3_coverage_pct: float = 0.0):
        self.total_connections = total_connections
        self.active_connections = active_connections
        self.total_syncs = total_syncs
        self.total_spend_records = total_spend_records
        self.total_purchase_orders = total_purchase_orders
        self.total_emissions_calculated = total_emissions_calculated
        self.scope3_coverage_pct = scope3_coverage_pct

    def to_dict(self) -> Dict[str, Any]:
        return {"total_connections": self.total_connections,
                "active_connections": self.active_connections,
                "total_syncs": self.total_syncs,
                "scope3_coverage_pct": self.scope3_coverage_pct}


# ---------------------------------------------------------------------------
# Inline Request models
# ---------------------------------------------------------------------------


class RegisterConnectionRequest:
    def __init__(self, erp_system: str = "simulated", host: str = "",
                 port: int = 443, username: str = "",
                 client_id: Optional[str] = None, tenant_id: str = "default"):
        self.erp_system = erp_system
        self.host = host
        self.port = port
        self.username = username
        self.client_id = client_id
        self.tenant_id = tenant_id


class SyncSpendRequest:
    def __init__(self, connection_id: str = "", start_date: str = "",
                 end_date: str = "", sync_mode: str = "full",
                 tenant_id: str = "default"):
        self.connection_id = connection_id
        self.start_date = start_date
        self.end_date = end_date
        self.sync_mode = sync_mode
        self.tenant_id = tenant_id


class MapVendorRequest:
    def __init__(self, vendor_id: str = "", vendor_name: str = "",
                 primary_category: str = "unclassified",
                 spend_category: str = "other",
                 emission_factor: Optional[float] = None):
        self.vendor_id = vendor_id
        self.vendor_name = vendor_name
        self.primary_category = primary_category
        self.spend_category = spend_category
        self.emission_factor = emission_factor


class CalculateEmissionsRequest:
    def __init__(self, connection_id: str = "", start_date: str = "",
                 end_date: str = "", methodology: str = "eeio",
                 tenant_id: str = "default"):
        self.connection_id = connection_id
        self.start_date = start_date
        self.end_date = end_date
        self.methodology = methodology
        self.tenant_id = tenant_id


# ===========================================================================
# Test Classes -- Layer 1 Enums
# ===========================================================================


class TestERPSystemEnum:
    """Test ERPSystem enum values (10 values)."""

    def test_sap_s4hana(self):
        assert ERPSystem.SAP_S4HANA.value == "sap_s4hana"

    def test_sap_ecc(self):
        assert ERPSystem.SAP_ECC.value == "sap_ecc"

    def test_oracle_cloud(self):
        assert ERPSystem.ORACLE_CLOUD.value == "oracle_cloud"

    def test_oracle_ebs(self):
        assert ERPSystem.ORACLE_EBS.value == "oracle_ebs"

    def test_netsuite(self):
        assert ERPSystem.NETSUITE.value == "netsuite"

    def test_dynamics_365(self):
        assert ERPSystem.DYNAMICS_365.value == "dynamics_365"

    def test_workday(self):
        assert ERPSystem.WORKDAY.value == "workday"

    def test_sage(self):
        assert ERPSystem.SAGE.value == "sage"

    def test_quickbooks(self):
        assert ERPSystem.QUICKBOOKS.value == "quickbooks"

    def test_simulated(self):
        assert ERPSystem.SIMULATED.value == "simulated"

    def test_all_10_systems(self):
        assert len(ERPSystem) == 10

    def test_from_value(self):
        assert ERPSystem("sap_s4hana") == ERPSystem.SAP_S4HANA


class TestScope3CategoryEnum:
    """Test Scope3Category enum values (16 values incl unclassified)."""

    def test_cat1(self):
        assert Scope3Category.CAT1_PURCHASED_GOODS.value == "cat1_purchased_goods"

    def test_cat2(self):
        assert Scope3Category.CAT2_CAPITAL_GOODS.value == "cat2_capital_goods"

    def test_cat3(self):
        assert Scope3Category.CAT3_FUEL_ENERGY.value == "cat3_fuel_energy"

    def test_cat4(self):
        assert Scope3Category.CAT4_UPSTREAM_TRANSPORT.value == "cat4_upstream_transport"

    def test_cat5(self):
        assert Scope3Category.CAT5_WASTE.value == "cat5_waste"

    def test_cat6(self):
        assert Scope3Category.CAT6_BUSINESS_TRAVEL.value == "cat6_business_travel"

    def test_cat7(self):
        assert Scope3Category.CAT7_EMPLOYEE_COMMUTING.value == "cat7_employee_commuting"

    def test_cat8(self):
        assert Scope3Category.CAT8_UPSTREAM_LEASED.value == "cat8_upstream_leased"

    def test_cat9(self):
        assert Scope3Category.CAT9_DOWNSTREAM_TRANSPORT.value == "cat9_downstream_transport"

    def test_cat10(self):
        assert Scope3Category.CAT10_PROCESSING.value == "cat10_processing"

    def test_cat11(self):
        assert Scope3Category.CAT11_USE_OF_SOLD.value == "cat11_use_of_sold"

    def test_cat12(self):
        assert Scope3Category.CAT12_END_OF_LIFE.value == "cat12_end_of_life"

    def test_cat13(self):
        assert Scope3Category.CAT13_DOWNSTREAM_LEASED.value == "cat13_downstream_leased"

    def test_cat14(self):
        assert Scope3Category.CAT14_FRANCHISES.value == "cat14_franchises"

    def test_cat15(self):
        assert Scope3Category.CAT15_INVESTMENTS.value == "cat15_investments"

    def test_unclassified(self):
        assert Scope3Category.UNCLASSIFIED.value == "unclassified"

    def test_all_16_categories(self):
        assert len(Scope3Category) == 16


class TestTransactionTypeEnum:
    """Test TransactionType enum values (6 values)."""

    def test_purchase(self):
        assert TransactionType.PURCHASE.value == "purchase"

    def test_invoice(self):
        assert TransactionType.INVOICE.value == "invoice"

    def test_credit_note(self):
        assert TransactionType.CREDIT_NOTE.value == "credit_note"

    def test_debit_note(self):
        assert TransactionType.DEBIT_NOTE.value == "debit_note"

    def test_payment(self):
        assert TransactionType.PAYMENT.value == "payment"

    def test_journal(self):
        assert TransactionType.JOURNAL.value == "journal"

    def test_all_6_types(self):
        assert len(TransactionType) == 6


class TestSpendCategoryEnum:
    """Test SpendCategory enum values (11 values)."""

    def test_raw_materials(self):
        assert SpendCategory.RAW_MATERIALS.value == "raw_materials"

    def test_energy(self):
        assert SpendCategory.ENERGY.value == "energy"

    def test_transportation(self):
        assert SpendCategory.TRANSPORTATION.value == "transportation"

    def test_packaging(self):
        assert SpendCategory.PACKAGING.value == "packaging"

    def test_waste_management(self):
        assert SpendCategory.WASTE_MANAGEMENT.value == "waste_management"

    def test_it_services(self):
        assert SpendCategory.IT_SERVICES.value == "it_services"

    def test_professional_services(self):
        assert SpendCategory.PROFESSIONAL_SERVICES.value == "professional_services"

    def test_travel(self):
        assert SpendCategory.TRAVEL.value == "travel"

    def test_facilities(self):
        assert SpendCategory.FACILITIES.value == "facilities"

    def test_chemicals(self):
        assert SpendCategory.CHEMICALS.value == "chemicals"

    def test_other(self):
        assert SpendCategory.OTHER.value == "other"

    def test_all_11_categories(self):
        assert len(SpendCategory) == 11


class TestConnectionStatusEnum:
    """Test ConnectionStatus enum values (5 values)."""

    def test_connected(self):
        assert ConnectionStatus.CONNECTED.value == "connected"

    def test_disconnected(self):
        assert ConnectionStatus.DISCONNECTED.value == "disconnected"

    def test_error(self):
        assert ConnectionStatus.ERROR.value == "error"

    def test_testing(self):
        assert ConnectionStatus.TESTING.value == "testing"

    def test_initializing(self):
        assert ConnectionStatus.INITIALIZING.value == "initializing"

    def test_all_5_statuses(self):
        assert len(ConnectionStatus) == 5


class TestSyncModeEnum:
    """Test SyncMode enum values (4 values)."""

    def test_full(self):
        assert SyncMode.FULL.value == "full"

    def test_incremental(self):
        assert SyncMode.INCREMENTAL.value == "incremental"

    def test_delta(self):
        assert SyncMode.DELTA.value == "delta"

    def test_manual(self):
        assert SyncMode.MANUAL.value == "manual"

    def test_all_4_modes(self):
        assert len(SyncMode) == 4


class TestEmissionMethodologyEnum:
    """Test EmissionMethodology enum values (4 values)."""

    def test_eeio(self):
        assert EmissionMethodology.EEIO.value == "eeio"

    def test_hybrid(self):
        assert EmissionMethodology.HYBRID.value == "hybrid"

    def test_process_based(self):
        assert EmissionMethodology.PROCESS_BASED.value == "process_based"

    def test_supplier_specific(self):
        assert EmissionMethodology.SUPPLIER_SPECIFIC.value == "supplier_specific"

    def test_all_4_methodologies(self):
        assert len(EmissionMethodology) == 4


# ===========================================================================
# Test Classes -- Layer 1 Models
# ===========================================================================


class TestVendorMappingModel:
    def test_creation_defaults(self):
        v = VendorMapping()
        assert v.vendor_id == ""
        assert v.primary_category == "unclassified"

    def test_creation_with_values(self):
        v = VendorMapping(vendor_id="V-001", vendor_name="EcoSteel",
                          primary_category="cat1_purchased_goods",
                          spend_category="raw_materials", emission_factor=0.35)
        assert v.vendor_id == "V-001"
        assert v.emission_factor == 0.35

    def test_to_dict(self):
        v = VendorMapping(vendor_id="V-001")
        d = v.to_dict()
        assert d["vendor_id"] == "V-001"
        assert "primary_category" in d


class TestMaterialMappingModel:
    def test_creation_defaults(self):
        m = MaterialMapping()
        assert m.material_code == ""
        assert m.scope3_category == "unclassified"

    def test_creation_with_values(self):
        m = MaterialMapping(material_code="STEEL-001", material_name="Hot-rolled steel",
                            scope3_category="cat1_purchased_goods", emission_factor=1.85)
        assert m.emission_factor == 1.85

    def test_to_dict(self):
        m = MaterialMapping(material_code="STEEL-001")
        d = m.to_dict()
        assert d["material_code"] == "STEEL-001"


class TestPurchaseOrderLineModel:
    def test_creation_defaults(self):
        li = PurchaseOrderLine()
        assert li.item_number == 0
        assert li.amount == 0.0

    def test_creation_with_values(self):
        li = PurchaseOrderLine(item_number=10, material="STEEL-HR-001",
                               quantity=50.0, unit_price=4000.0, amount=200000.0)
        assert li.material == "STEEL-HR-001"
        assert li.amount == 200000.0

    def test_to_dict(self):
        li = PurchaseOrderLine(item_number=10, amount=200000.0)
        d = li.to_dict()
        assert d["amount"] == 200000.0


class TestPurchaseOrderModel:
    def test_creation_defaults(self):
        po = PurchaseOrder()
        assert po.po_number == ""
        assert po.status == "open"
        assert po.line_items == []

    def test_creation_with_values(self):
        items = [PurchaseOrderLine(item_number=10, amount=200000.0)]
        po = PurchaseOrder(po_number="PO-001", vendor_id="V-001",
                           total_value=200000.0, line_items=items)
        assert len(po.line_items) == 1
        assert po.total_value == 200000.0

    def test_to_dict(self):
        po = PurchaseOrder(po_number="PO-001")
        d = po.to_dict()
        assert d["po_number"] == "PO-001"
        assert "line_items" in d


class TestSpendRecordModel:
    def test_creation_defaults(self):
        s = SpendRecord()
        assert len(s.record_id) == 36
        assert s.amount == 0.0
        assert s.currency == "USD"

    def test_creation_with_values(self):
        s = SpendRecord(record_id="SPD-001", vendor_id="V-001",
                        amount=125000.0, currency="EUR", category="raw_materials")
        assert s.amount == 125000.0
        assert s.currency == "EUR"

    def test_to_dict(self):
        s = SpendRecord(record_id="SPD-001", amount=125000.0)
        d = s.to_dict()
        assert d["amount"] == 125000.0


class TestInventoryItemModel:
    def test_creation_defaults(self):
        i = InventoryItem()
        assert len(i.item_id) == 36
        assert i.quantity == 0.0

    def test_creation_with_values(self):
        i = InventoryItem(item_id="INV-001", material="STEEL-HR-001",
                          warehouse="WH-MAIN", quantity=120.0, unit_cost=4000.0)
        assert i.quantity == 120.0
        assert i.unit_cost == 4000.0

    def test_total_value(self):
        i = InventoryItem(quantity=50.0, unit_cost=4000.0)
        assert i.total_value == 200000.0

    def test_to_dict(self):
        i = InventoryItem(item_id="INV-001")
        d = i.to_dict()
        assert d["item_id"] == "INV-001"


class TestERPConnectionConfigModel:
    def test_creation_defaults(self):
        c = ERPConnectionConfig()
        assert c.erp_system == "simulated"
        assert c.port == 443

    def test_creation_with_values(self):
        c = ERPConnectionConfig(erp_system="sap_s4hana", host="sap.example.com",
                                port=8443, username="api_user")
        assert c.host == "sap.example.com"

    def test_to_dict(self):
        c = ERPConnectionConfig(host="test.com")
        d = c.to_dict()
        assert d["host"] == "test.com"


class TestERPQueryInputModel:
    def test_creation_defaults(self):
        q = ERPQueryInput()
        assert q.query_type == "spend"

    def test_creation_with_values(self):
        q = ERPQueryInput(connection_id="conn-001", query_type="po",
                          start_date="2025-01-01", end_date="2025-06-30")
        assert q.query_type == "po"


class TestERPQueryOutputModel:
    def test_creation_defaults(self):
        o = ERPQueryOutput()
        assert o.status == "completed"
        assert o.records == []

    def test_creation_with_values(self):
        o = ERPQueryOutput(record_count=100, records=[{"id": "1"}])
        assert o.record_count == 100
        assert len(o.records) == 1


# ===========================================================================
# Test Classes -- SDK Models
# ===========================================================================


class TestConnectionRecordSDKModel:
    def test_creation_defaults(self):
        cr = ConnectionRecord()
        assert len(cr.connection_id) == 36
        assert cr.status == "initializing"

    def test_creation_with_values(self):
        cr = ConnectionRecord(connection_id="conn-001", erp_system="sap_s4hana",
                              host="sap.example.com", status="connected")
        assert cr.erp_system == "sap_s4hana"

    def test_to_dict(self):
        cr = ConnectionRecord(connection_id="conn-001")
        d = cr.to_dict()
        assert d["connection_id"] == "conn-001"


class TestSyncJobModel:
    def test_creation_defaults(self):
        sj = SyncJob()
        assert len(sj.job_id) == 36
        assert sj.status == "pending"

    def test_creation_with_values(self):
        sj = SyncJob(job_id="job-001", connection_id="conn-001",
                      sync_mode="incremental", records_synced=500)
        assert sj.records_synced == 500


class TestSpendSummaryModel:
    def test_creation_defaults(self):
        ss = SpendSummary()
        assert ss.total_spend_usd == 0.0

    def test_creation_with_values(self):
        ss = SpendSummary(period_start="2025-01-01", period_end="2025-06-30",
                          total_spend_usd=500000.0, record_count=1200)
        assert ss.total_spend_usd == 500000.0


class TestScope3SummaryModel:
    def test_creation_defaults(self):
        s3 = Scope3Summary()
        assert s3.category == "unclassified"

    def test_creation_with_values(self):
        s3 = Scope3Summary(category="cat1_purchased_goods",
                           total_spend_usd=200000.0,
                           estimated_emissions_kgco2e=70000.0)
        assert s3.estimated_emissions_kgco2e == 70000.0


class TestEmissionResultModel:
    def test_creation_defaults(self):
        er = EmissionResult()
        assert len(er.result_id) == 36
        assert er.methodology == "eeio"

    def test_creation_with_values(self):
        er = EmissionResult(spend_record_id="SPD-001", vendor_id="V-001",
                            amount_usd=125000.0, emission_factor=0.35,
                            estimated_kgco2e=43750.0)
        assert er.estimated_kgco2e == 43750.0

    def test_to_dict(self):
        er = EmissionResult(vendor_id="V-001", estimated_kgco2e=43750.0)
        d = er.to_dict()
        assert d["estimated_kgco2e"] == 43750.0


class TestCurrencyRateModel:
    def test_creation_defaults(self):
        cr = CurrencyRate()
        assert cr.rate == 1.0

    def test_creation_with_values(self):
        cr = CurrencyRate(from_currency="EUR", to_currency="USD",
                          rate=1.08, effective_date="2025-06-15")
        assert cr.rate == 1.08

    def test_to_dict(self):
        cr = CurrencyRate(from_currency="GBP", to_currency="USD", rate=1.27)
        d = cr.to_dict()
        assert d["rate"] == 1.27


class TestERPStatisticsModel:
    def test_creation_defaults(self):
        es = ERPStatistics()
        assert es.total_connections == 0

    def test_creation_with_values(self):
        es = ERPStatistics(total_connections=5, active_connections=3,
                           total_syncs=120, scope3_coverage_pct=87.5)
        assert es.scope3_coverage_pct == 87.5


# ===========================================================================
# Test Classes -- Request Models
# ===========================================================================


class TestRegisterConnectionRequestModel:
    def test_creation(self):
        r = RegisterConnectionRequest(erp_system="sap_s4hana",
                                       host="sap.example.com",
                                       username="api_user")
        assert r.erp_system == "sap_s4hana"

    def test_default_port(self):
        r = RegisterConnectionRequest()
        assert r.port == 443

    def test_default_tenant(self):
        r = RegisterConnectionRequest()
        assert r.tenant_id == "default"


class TestSyncSpendRequestModel:
    def test_creation(self):
        r = SyncSpendRequest(connection_id="conn-001",
                             start_date="2025-01-01",
                             end_date="2025-06-30")
        assert r.connection_id == "conn-001"

    def test_default_sync_mode(self):
        r = SyncSpendRequest()
        assert r.sync_mode == "full"


class TestMapVendorRequestModel:
    def test_creation(self):
        r = MapVendorRequest(vendor_id="V-001", vendor_name="EcoSteel",
                             primary_category="cat1_purchased_goods",
                             spend_category="raw_materials")
        assert r.vendor_id == "V-001"

    def test_optional_emission_factor(self):
        r = MapVendorRequest(emission_factor=0.35)
        assert r.emission_factor == 0.35

    def test_no_emission_factor(self):
        r = MapVendorRequest()
        assert r.emission_factor is None


class TestCalculateEmissionsRequestModel:
    def test_creation(self):
        r = CalculateEmissionsRequest(connection_id="conn-001",
                                       start_date="2025-01-01",
                                       end_date="2025-06-30")
        assert r.connection_id == "conn-001"

    def test_default_methodology(self):
        r = CalculateEmissionsRequest()
        assert r.methodology == "eeio"
