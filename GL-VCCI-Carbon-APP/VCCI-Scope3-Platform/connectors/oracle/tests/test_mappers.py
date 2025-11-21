# -*- coding: utf-8 -*-
"""
Oracle Connector Mappers Tests
GL-VCCI Scope 3 Platform

Tests for data mappers including PO, Requisition, Shipment,
and Transport mappers with unit conversion and validation.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Test Count: 30
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from connectors.oracle.mappers.po_mapper import PurchaseOrderMapper, ProcurementRecord
from connectors.oracle.mappers.requisition_mapper import RequisitionMapper
from connectors.oracle.mappers.shipment_mapper import ShipmentMapper
from connectors.oracle.mappers.transport_mapper import TransportMapper
from greenlang.determinism import DeterministicClock


class TestPurchaseOrderMapper:
    """Tests for PurchaseOrderMapper."""

    @pytest.fixture
    def po_mapper(self):
        """Create PO mapper instance."""
        return PurchaseOrderMapper(tenant_id="test_tenant")

    def test_mapper_initialization(self, po_mapper):
        """Test PO mapper initialization."""
        assert po_mapper.tenant_id == "test_tenant"

    def test_standardize_unit_known_units(self, po_mapper):
        """Test unit standardization for known units."""
        assert po_mapper._standardize_unit("KG") == "kg"
        assert po_mapper._standardize_unit("EA") == "items"
        assert po_mapper._standardize_unit("LTR") == "liters"
        assert po_mapper._standardize_unit("TON") == "tonnes"
        assert po_mapper._standardize_unit("KWH") == "kWh"

    def test_standardize_unit_unknown_unit(self, po_mapper):
        """Test unit standardization for unknown units."""
        assert po_mapper._standardize_unit("UNKNOWN") == "units"
        assert po_mapper._standardize_unit(None) == "units"
        assert po_mapper._standardize_unit("") == "units"

    def test_convert_currency_usd_to_usd(self, po_mapper):
        """Test currency conversion USD to USD."""
        amount_usd, rate = po_mapper._convert_currency(1000.0, "USD")

        assert amount_usd == 1000.0
        assert rate == 1.0

    def test_convert_currency_eur_to_usd(self, po_mapper):
        """Test currency conversion EUR to USD."""
        amount_usd, rate = po_mapper._convert_currency(1000.0, "EUR")

        assert amount_usd == 1100.0  # Using static rate 1.10
        assert rate == 1.10

    def test_convert_currency_gbp_to_usd(self, po_mapper):
        """Test currency conversion GBP to USD."""
        amount_usd, rate = po_mapper._convert_currency(1000.0, "GBP")

        assert amount_usd == 1270.0  # Using static rate 1.27
        assert rate == 1.27

    def test_convert_currency_unknown_currency(self, po_mapper):
        """Test currency conversion for unknown currency."""
        amount_usd, rate = po_mapper._convert_currency(1000.0, "XYZ")

        assert amount_usd == 1000.0  # Default to 1.0 rate
        assert rate == 1.0

    def test_generate_procurement_id(self, po_mapper):
        """Test procurement ID generation."""
        proc_id = po_mapper._generate_procurement_id(123456789, 1)

        assert proc_id == "PROC-123456789-00001"

    def test_generate_procurement_id_padding(self, po_mapper):
        """Test procurement ID generation with line number padding."""
        proc_id = po_mapper._generate_procurement_id(123, 99999)

        assert proc_id == "PROC-123-99999"

    def test_extract_reporting_year(self, po_mapper):
        """Test extracting reporting year from date."""
        year = po_mapper._extract_reporting_year("2024-03-15")

        assert year == 2024

    def test_extract_reporting_year_invalid_date(self, po_mapper):
        """Test extracting reporting year with invalid date."""
        year = po_mapper._extract_reporting_year("invalid-date")

        # Should return current year
        assert year == DeterministicClock.now().year

    def test_map_purchase_order_basic(self, po_mapper, sample_purchase_order, sample_po_line):
        """Test basic PO mapping."""
        record = po_mapper.map_purchase_order(sample_purchase_order, sample_po_line)

        assert isinstance(record, ProcurementRecord)
        assert record.procurement_id == "PROC-300100123456789-00001"
        assert record.supplier_name == "Acme Corp"
        assert record.product_name == "Widget Type A"
        assert record.quantity == 100.0
        assert record.unit == "items"
        assert record.spend_usd == 5000.0

    def test_map_purchase_order_with_supplier_data(self, po_mapper, sample_purchase_order, sample_po_line, sample_supplier):
        """Test PO mapping with supplier enrichment."""
        record = po_mapper.map_purchase_order(
            sample_purchase_order,
            sample_po_line,
            supplier_data=sample_supplier
        )

        assert record.supplier_country == "US"
        assert record.supplier_region == "US"

    def test_map_purchase_order_missing_required_fields(self, po_mapper, sample_po_line):
        """Test PO mapping with missing required fields."""
        invalid_header = {}  # Missing POHeaderId

        with pytest.raises(ValueError, match="Missing required field"):
            po_mapper.map_purchase_order(invalid_header, sample_po_line)

    def test_map_purchase_order_missing_supplier_name(self, po_mapper, sample_po_line):
        """Test PO mapping with missing supplier name."""
        header = {
            "POHeaderId": 123,
            "SupplierId": 1001,
            "OrderedDate": "2024-01-15",
            "Currency": "USD"
        }

        record = po_mapper.map_purchase_order(header, sample_po_line)

        # Should use supplier ID as fallback
        assert record.supplier_name == "Supplier 1001"

    def test_map_purchase_order_currency_conversion(self, po_mapper, sample_po_line):
        """Test PO mapping with currency conversion."""
        header = {
            "POHeaderId": 123,
            "OrderNumber": "PO-001",
            "SupplierId": 1001,
            "SupplierName": "Test Supplier",
            "OrderedDate": "2024-01-15",
            "Currency": "EUR"
        }

        record = po_mapper.map_purchase_order(header, sample_po_line)

        assert record.spend_currency_original == "EUR"
        assert record.spend_amount_original == 5000.0
        assert record.spend_usd == 5500.0  # 5000 * 1.10
        assert record.exchange_rate_to_usd == 1.10

    def test_map_purchase_order_metadata(self, po_mapper, sample_purchase_order, sample_po_line):
        """Test PO mapping includes metadata."""
        record = po_mapper.map_purchase_order(sample_purchase_order, sample_po_line)

        assert record.metadata is not None
        assert record.metadata["source_system"] == "Oracle_Fusion"
        assert "extraction_timestamp" in record.metadata
        assert "validation_status" in record.metadata

    def test_map_purchase_order_custom_fields(self, po_mapper, sample_purchase_order, sample_po_line):
        """Test PO mapping includes custom fields."""
        record = po_mapper.map_purchase_order(sample_purchase_order, sample_po_line)

        assert record.custom_fields is not None
        assert record.custom_fields["po_header_id"] == 300100123456789
        assert record.custom_fields["po_number"] == "PO-2024-001234"
        assert record.custom_fields["business_unit"] == "US01"

    def test_map_batch(self, po_mapper, sample_purchase_order, sample_po_line):
        """Test batch mapping of POs."""
        po_data = [
            {"header": sample_purchase_order, "line": sample_po_line},
            {"header": {**sample_purchase_order, "POHeaderId": 999}, "line": {**sample_po_line, "LineNumber": 2}}
        ]

        records = po_mapper.map_batch(po_data)

        assert len(records) == 2
        assert records[0].procurement_id == "PROC-300100123456789-00001"
        assert records[1].procurement_id == "PROC-999-00002"

    def test_map_batch_with_errors(self, po_mapper, sample_po_line):
        """Test batch mapping with invalid records."""
        po_data = [
            {"header": {}, "line": sample_po_line},  # Invalid - missing POHeaderId
            {"header": {"POHeaderId": 123, "OrderedDate": "2024-01-01", "Currency": "USD"}, "line": sample_po_line}
        ]

        records = po_mapper.map_batch(po_data)

        # Should skip invalid record
        assert len(records) == 1


class TestRequisitionMapper:
    """Tests for RequisitionMapper."""

    @pytest.fixture
    def req_mapper(self):
        """Create Requisition mapper instance."""
        return RequisitionMapper(tenant_id="test_tenant")

    def test_mapper_initialization(self, req_mapper):
        """Test requisition mapper initialization."""
        assert req_mapper.tenant_id == "test_tenant"

    def test_map_requisition_basic(self, req_mapper, sample_requisition):
        """Test basic requisition mapping."""
        from connectors.oracle.mappers.requisition_mapper import RequisitionRecord

        record = req_mapper.map_requisition(sample_requisition, {})

        assert isinstance(record, RequisitionRecord)
        assert record.procurement_id.startswith("REQ-")

    def test_map_requisition_with_lines(self, req_mapper, sample_requisition):
        """Test requisition mapping with line items."""
        line = {
            "LineNumber": 1,
            "ItemDescription": "Office Chair",
            "Quantity": 5,
            "UOM": "EA",
            "LineAmount": 500.0
        }

        record = req_mapper.map_requisition(sample_requisition, line)

        assert record.product_name == "Office Chair"
        assert record.quantity == 5.0


class TestShipmentMapper:
    """Tests for ShipmentMapper."""

    @pytest.fixture
    def shipment_mapper(self):
        """Create Shipment mapper instance."""
        return ShipmentMapper(tenant_id="test_tenant")

    def test_mapper_initialization(self, shipment_mapper):
        """Test shipment mapper initialization."""
        assert shipment_mapper.tenant_id == "test_tenant"

    def test_map_shipment_basic(self, shipment_mapper, sample_shipment):
        """Test basic shipment mapping."""
        from connectors.oracle.mappers.shipment_mapper import LogisticsRecord

        record = shipment_mapper.map_shipment(sample_shipment)

        assert isinstance(record, LogisticsRecord)
        assert record.logistics_id.startswith("SHIP-")

    def test_map_shipment_weight_conversion(self, shipment_mapper):
        """Test shipment weight conversion."""
        shipment = {
            "ShipmentId": 123,
            "ShipmentNumber": "SHIP-001",
            "ShipFromLocation": "WH-A",
            "ShipToLocation": "WH-B",
            "ShipmentDate": "2024-01-20",
            "TotalWeight": 1000.0,
            "WeightUOM": "LB",
            "Status": "DELIVERED"
        }

        record = shipment_mapper.map_shipment(shipment)

        # Should convert LB to kg
        assert record.weight_kg > 0


class TestTransportMapper:
    """Tests for TransportMapper."""

    @pytest.fixture
    def transport_mapper(self):
        """Create Transport mapper instance."""
        return TransportMapper(tenant_id="test_tenant")

    def test_mapper_initialization(self, transport_mapper):
        """Test transport mapper initialization."""
        assert transport_mapper.tenant_id == "test_tenant"

    def test_map_transport_order_basic(self, transport_mapper, sample_transport_order):
        """Test basic transport order mapping."""
        from connectors.oracle.mappers.transport_mapper import TransportRecord

        record = transport_mapper.map_transport_order(sample_transport_order)

        assert isinstance(record, TransportRecord)
        assert record.transport_id.startswith("TRANS-")

    def test_map_transport_mode_standardization(self, transport_mapper):
        """Test transport mode standardization."""
        transport_order = {
            "TransportOrderId": 123,
            "OrderNumber": "TRANS-001",
            "OriginLocation": "NYC",
            "DestinationLocation": "LA",
            "TransportMode": "TRUCK",
            "Distance": 2800.0,
            "DistanceUOM": "MI",
            "Weight": 10000.0,
            "WeightUOM": "LB",
            "PickupDate": "2024-02-01",
            "DeliveryDate": "2024-02-05",
            "Status": "IN_TRANSIT"
        }

        record = transport_mapper.map_transport_order(transport_order)

        assert record.transport_mode in ["road", "rail", "air", "sea"]

    def test_map_transport_distance_conversion(self, transport_mapper, sample_transport_order):
        """Test distance conversion from miles to km."""
        record = transport_mapper.map_transport_order(sample_transport_order)

        # 2800 miles should be converted to km (~4500 km)
        assert record.distance_km > 4000

    def test_map_transport_missing_distance(self, transport_mapper):
        """Test transport mapping with missing distance."""
        transport_order = {
            "TransportOrderId": 123,
            "OrderNumber": "TRANS-001",
            "OriginLocation": "NYC",
            "DestinationLocation": "LA",
            "TransportMode": "TRUCK",
            "PickupDate": "2024-02-01",
            "DeliveryDate": "2024-02-05",
            "Status": "IN_TRANSIT"
        }

        record = transport_mapper.map_transport_order(transport_order)

        # Should handle missing distance gracefully
        assert record.distance_km is None or record.distance_km == 0

    def test_map_transport_region_inference(self, transport_mapper):
        """Test region inference from locations."""
        transport_order = {
            "TransportOrderId": 123,
            "OrderNumber": "TRANS-001",
            "OriginLocation": "NYC-WH",
            "DestinationLocation": "LA-WH",
            "TransportMode": "TRUCK",
            "Distance": 2800.0,
            "DistanceUOM": "MI",
            "Weight": 10000.0,
            "WeightUOM": "LB",
            "PickupDate": "2024-02-01",
            "DeliveryDate": "2024-02-05",
            "Status": "IN_TRANSIT"
        }

        record = transport_mapper.map_transport_order(transport_order)

        # Should infer US region
        assert record.origin_region is not None or record.destination_region is not None
