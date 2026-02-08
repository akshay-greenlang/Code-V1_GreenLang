# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for ERP/Finance Connector (AGENT-DATA-003)

Tests full ERP connector lifecycle: connection registration -> spend sync ->
Scope 3 mapping -> emissions calculation -> provenance verification.
Tests multi-currency normalization, vendor mapping overrides, PO sync with
goods receipt matching, and error handling across the pipeline.

All implementations are self-contained to avoid cross-module import issues.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Self-contained implementations for integration testing
# ---------------------------------------------------------------------------


DEFAULT_RATES_TO_USD = {
    "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067,
    "CHF": 1.13, "CAD": 0.74, "AUD": 0.65, "CNY": 0.14,
    "INR": 0.012, "BRL": 0.20,
}

DEFAULT_EEIO_FACTORS = {
    "cat1_purchased_goods": 0.35,
    "cat2_capital_goods": 0.42,
    "cat3_fuel_energy": 0.55,
    "cat4_upstream_transport": 0.28,
    "cat5_waste": 0.18,
    "cat6_business_travel": 0.22,
    "cat7_employee_commuting": 0.15,
    "cat8_upstream_leased": 0.30,
    "cat9_downstream_transport": 0.25,
    "cat10_processing": 0.38,
    "cat11_use_of_products": 0.45,
    "cat12_end_of_life": 0.20,
    "cat13_downstream_leased": 0.32,
    "cat14_franchises": 0.28,
    "cat15_investments": 0.40,
    "unclassified": 0.50,
}

SPEND_TO_SCOPE3 = {
    "raw_materials": "cat1_purchased_goods",
    "packaging": "cat1_purchased_goods",
    "services": "cat1_purchased_goods",
    "capital_equipment": "cat2_capital_goods",
    "fuel": "cat3_fuel_energy",
    "electricity": "cat3_fuel_energy",
    "logistics": "cat4_upstream_transport",
    "waste_management": "cat5_waste",
    "travel": "cat6_business_travel",
    "commuting": "cat7_employee_commuting",
}


class ERPConnectorPipeline:
    """End-to-end ERP connector pipeline for integration testing."""

    GENESIS_HASH = "0" * 64

    def __init__(self):
        self._connections: Dict[str, Dict[str, Any]] = {}
        self._spend_records: List[Dict[str, Any]] = []
        self._purchase_orders: Dict[str, Dict[str, Any]] = {}
        self._inventory: Dict[str, Dict[str, Any]] = {}
        self._vendor_mappings: Dict[str, Dict[str, Any]] = {}
        self._emission_results: List[Dict[str, Any]] = []
        self._provenance: Dict[str, List[Dict[str, Any]]] = {}
        self._rates = dict(DEFAULT_RATES_TO_USD)

    # --- Connection Management ---

    def register_connection(self, erp_system: str, host: str,
                            username: str = "api_user") -> Dict[str, Any]:
        conn_id = f"conn-{hashlib.sha256(f'{erp_system}:{host}'.encode()).hexdigest()[:12]}"
        record = {
            "connection_id": conn_id,
            "erp_system": erp_system,
            "host": host,
            "username": username,
            "status": "registered",
            "created_at": datetime.utcnow().isoformat(),
        }
        self._connections[conn_id] = record
        self._record_provenance(conn_id, "register_connection",
                                {"erp_system": erp_system, "host": host})
        return record

    def test_connection(self, connection_id: str) -> Dict[str, Any]:
        if connection_id not in self._connections:
            raise ValueError(f"Unknown connection: {connection_id}")
        self._connections[connection_id]["status"] = "connected"
        self._record_provenance(connection_id, "test_connection",
                                {"result": "success"})
        return {"success": True, "connection_id": connection_id}

    # --- Spend Sync ---

    def sync_spend(self, connection_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if connection_id not in self._connections:
            raise ValueError(f"Unknown connection: {connection_id}")
        for r in records:
            r.setdefault("record_id", f"SPD-{uuid.uuid4().hex[:8]}")
            r.setdefault("connection_id", connection_id)
        self._spend_records.extend(records)
        self._record_provenance(connection_id, "sync_spend",
                                {"records_synced": len(records)})
        return {"records_synced": len(records), "connection_id": connection_id}

    # --- Currency Conversion ---

    def convert_currency(self, amount: float, from_currency: str,
                         to_currency: str = "USD") -> float:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        if from_currency == to_currency:
            return round(amount, 2)
        from_rate = self._rates.get(from_currency)
        to_rate = self._rates.get(to_currency)
        if from_rate is None:
            raise ValueError(f"Unsupported currency: {from_currency}")
        if to_rate is None:
            raise ValueError(f"Unsupported currency: {to_currency}")
        return round((amount * from_rate) / to_rate, 2)

    def normalize_spend_currency(self, to_currency: str = "USD") -> List[Dict[str, Any]]:
        """Convert all spend records to a single currency."""
        normalized = []
        for r in self._spend_records:
            amount = r.get("amount", 0.0)
            from_cur = r.get("currency", "USD")
            converted = self.convert_currency(amount, from_cur, to_currency)
            normalized_record = {**r, "amount_normalized": converted,
                                 "normalized_currency": to_currency}
            normalized.append(normalized_record)
        return normalized

    # --- Purchase Order Sync ---

    def sync_purchase_orders(self, connection_id: str,
                             orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        if connection_id not in self._connections:
            raise ValueError(f"Unknown connection: {connection_id}")
        for po in orders:
            po_number = po.get("po_number", f"PO-{uuid.uuid4().hex[:8]}")
            po["po_number"] = po_number
            po["connection_id"] = connection_id
            self._purchase_orders[po_number] = po
        self._record_provenance(connection_id, "sync_purchase_orders",
                                {"orders_synced": len(orders)})
        return {"orders_synced": len(orders)}

    def match_goods_receipt(self, po_number: str, received_qty: float) -> Dict[str, Any]:
        """Match goods receipt against PO line items."""
        po = self._purchase_orders.get(po_number)
        if not po:
            raise ValueError(f"Unknown PO: {po_number}")
        ordered_qty = sum(
            li.get("quantity", 0.0) for li in po.get("line_items", [])
        )
        variance_pct = 0.0
        if ordered_qty > 0:
            variance_pct = round(((received_qty - ordered_qty) / ordered_qty) * 100, 2)
        status = "full_match" if abs(variance_pct) < 1.0 else (
            "partial_match" if received_qty < ordered_qty else "over_receipt"
        )
        return {
            "po_number": po_number,
            "ordered_qty": ordered_qty,
            "received_qty": received_qty,
            "variance_pct": variance_pct,
            "status": status,
        }

    # --- Scope 3 Mapping ---

    def classify_spend(self, spend_category: str,
                       vendor_id: Optional[str] = None) -> str:
        """Classify spend into Scope 3 category with vendor override."""
        if vendor_id and vendor_id in self._vendor_mappings:
            return self._vendor_mappings[vendor_id].get(
                "scope3_category", "unclassified"
            )
        return SPEND_TO_SCOPE3.get(spend_category, "unclassified")

    def map_vendor(self, vendor_id: str, vendor_name: str,
                   scope3_category: str) -> Dict[str, Any]:
        mapping = {
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "scope3_category": scope3_category,
        }
        self._vendor_mappings[vendor_id] = mapping
        return mapping

    def classify_all_spend(self) -> List[Dict[str, Any]]:
        """Classify all spend records into Scope 3 categories."""
        classified = []
        for r in self._spend_records:
            scope3 = self.classify_spend(
                r.get("category", ""),
                r.get("vendor_id"),
            )
            classified.append({**r, "scope3_category": scope3})
        return classified

    # --- Emissions Calculation ---

    def calculate_emissions(self, classified_spend: Optional[List[Dict[str, Any]]] = None,
                            methodology: str = "eeio") -> Dict[str, Any]:
        if classified_spend is None:
            classified_spend = self.classify_all_spend()

        results = []
        for r in classified_spend:
            scope3 = r.get("scope3_category", "unclassified")
            amount = r.get("amount_normalized", r.get("amount", 0.0))
            factor = DEFAULT_EEIO_FACTORS.get(scope3, 0.50)
            kgco2e = round(amount * factor, 2)
            results.append({
                "record_id": r.get("record_id", ""),
                "scope3_category": scope3,
                "amount_usd": amount,
                "emission_factor": factor,
                "estimated_kgco2e": kgco2e,
                "methodology": methodology,
            })
        self._emission_results.extend(results)
        total = sum(r["estimated_kgco2e"] for r in results)
        return {
            "total_emissions_kgco2e": round(total, 2),
            "records_calculated": len(results),
            "methodology": methodology,
        }

    def get_emissions_by_category(self) -> Dict[str, float]:
        by_cat: Dict[str, float] = {}
        for r in self._emission_results:
            cat = r.get("scope3_category", "unclassified")
            by_cat[cat] = round(by_cat.get(cat, 0.0) + r["estimated_kgco2e"], 2)
        return by_cat

    # --- Provenance ---

    def _record_provenance(self, chain_id: str, operation: str,
                           data: Dict[str, Any]):
        if chain_id not in self._provenance:
            self._provenance[chain_id] = []
        chain = self._provenance[chain_id]
        prev_hash = chain[-1]["hash"] if chain else self.GENESIS_HASH
        record_data = json.dumps({"op": operation, "data": data}, sort_keys=True)
        record_hash = hashlib.sha256(record_data.encode()).hexdigest()
        chain.append({
            "sequence": len(chain) + 1,
            "operation": operation,
            "data": data,
            "previous_hash": prev_hash,
            "hash": record_hash,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_provenance_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        return list(self._provenance.get(chain_id, []))

    def verify_provenance(self, chain_id: str) -> bool:
        chain = self._provenance.get(chain_id, [])
        if not chain:
            return True
        for i, record in enumerate(chain):
            expected_prev = chain[i - 1]["hash"] if i > 0 else self.GENESIS_HASH
            if record["previous_hash"] != expected_prev:
                return False
        return True

    # --- Statistics ---

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_connections": len(self._connections),
            "total_spend_records": len(self._spend_records),
            "total_purchase_orders": len(self._purchase_orders),
            "total_inventory_items": len(self._inventory),
            "total_vendor_mappings": len(self._vendor_mappings),
            "total_emission_results": len(self._emission_results),
        }


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------


SAMPLE_SPEND_RECORDS_MULTI_CURRENCY = [
    {"record_id": "SPD-001", "vendor_id": "V-001", "amount": 125000.0,
     "currency": "EUR", "category": "raw_materials", "vendor_name": "EcoSteel GmbH"},
    {"record_id": "SPD-002", "vendor_id": "V-002", "amount": 78500.0,
     "currency": "USD", "category": "logistics", "vendor_name": "GreenFreight Inc"},
    {"record_id": "SPD-003", "vendor_id": "V-003", "amount": 9500000.0,
     "currency": "JPY", "category": "packaging", "vendor_name": "PacificPack KK"},
    {"record_id": "SPD-004", "vendor_id": "V-004", "amount": 45000.0,
     "currency": "GBP", "category": "services", "vendor_name": "AuditPartners LLP"},
    {"record_id": "SPD-005", "vendor_id": "V-005", "amount": 200000.0,
     "currency": "CNY", "category": "capital_equipment", "vendor_name": "ShenZhen Mfg"},
]

SAMPLE_PURCHASE_ORDERS = [
    {
        "po_number": "PO-2025-0001",
        "vendor_id": "V-001",
        "vendor_name": "EcoSteel GmbH",
        "status": "open",
        "total_value": 125000.0,
        "currency": "EUR",
        "line_items": [
            {"item_id": "LI-001", "material": "Steel Plate A36", "quantity": 500.0,
             "unit": "kg", "unit_price": 150.0},
            {"item_id": "LI-002", "material": "Steel Rod 1045", "quantity": 200.0,
             "unit": "kg", "unit_price": 250.0},
        ],
    },
    {
        "po_number": "PO-2025-0002",
        "vendor_id": "V-002",
        "vendor_name": "GreenFreight Inc",
        "status": "open",
        "total_value": 78500.0,
        "currency": "USD",
        "line_items": [
            {"item_id": "LI-003", "material": "Freight Service", "quantity": 10.0,
             "unit": "shipment", "unit_price": 7850.0},
        ],
    },
]


# ===========================================================================
# Test Classes
# ===========================================================================


class TestConnectionLifecycle:
    """Test full connection lifecycle: register -> test -> use -> verify."""

    def test_full_connection_lifecycle(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("sap_s4hana", "sap.example.com")
        assert conn["status"] == "registered"
        assert conn["connection_id"].startswith("conn-")

        test_result = pipeline.test_connection(conn["connection_id"])
        assert test_result["success"] is True

        # Verify provenance was recorded
        chain = pipeline.get_provenance_chain(conn["connection_id"])
        assert len(chain) == 2
        assert chain[0]["operation"] == "register_connection"
        assert chain[1]["operation"] == "test_connection"

    def test_connection_provenance_integrity(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("oracle_cloud", "oracle.example.com")
        pipeline.test_connection(conn["connection_id"])
        assert pipeline.verify_provenance(conn["connection_id"]) is True

    def test_unknown_connection_raises(self):
        pipeline = ERPConnectorPipeline()
        with pytest.raises(ValueError, match="Unknown connection"):
            pipeline.test_connection("conn-nonexistent")


class TestSpendSyncAndScope3Pipeline:
    """Test spend sync -> Scope 3 classification -> emissions calculation."""

    def test_full_spend_to_emissions_pipeline(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")
        pipeline.test_connection(conn["connection_id"])

        # Sync spend records
        sync_result = pipeline.sync_spend(
            conn["connection_id"], SAMPLE_SPEND_RECORDS_MULTI_CURRENCY
        )
        assert sync_result["records_synced"] == 5

        # Normalize currency
        normalized = pipeline.normalize_spend_currency("USD")
        assert len(normalized) == 5
        for r in normalized:
            assert "amount_normalized" in r
            assert r["normalized_currency"] == "USD"

        # Classify into Scope 3
        classified = pipeline.classify_all_spend()
        assert len(classified) == 5
        categories = {r["scope3_category"] for r in classified}
        assert "cat1_purchased_goods" in categories
        assert "cat4_upstream_transport" in categories

        # Calculate emissions
        emissions = pipeline.calculate_emissions(classified)
        assert emissions["total_emissions_kgco2e"] > 0
        assert emissions["records_calculated"] == 5

    def test_scope3_classification_with_vendor_override(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")

        # Map vendor to override default classification
        pipeline.map_vendor("V-002", "GreenFreight Inc", "cat9_downstream_transport")

        # Sync spend
        pipeline.sync_spend(conn["connection_id"], [
            {"record_id": "SPD-X1", "vendor_id": "V-002", "amount": 50000.0,
             "currency": "USD", "category": "logistics"},
        ])

        # Classify - vendor override should take precedence
        classified = pipeline.classify_all_spend()
        assert classified[0]["scope3_category"] == "cat9_downstream_transport"

    def test_emissions_by_category_breakdown(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")
        pipeline.sync_spend(conn["connection_id"], [
            {"record_id": "S1", "amount": 100000.0, "currency": "USD",
             "category": "raw_materials"},
            {"record_id": "S2", "amount": 50000.0, "currency": "USD",
             "category": "fuel"},
            {"record_id": "S3", "amount": 30000.0, "currency": "USD",
             "category": "logistics"},
        ])
        pipeline.calculate_emissions()
        by_cat = pipeline.get_emissions_by_category()
        assert "cat1_purchased_goods" in by_cat
        assert "cat3_fuel_energy" in by_cat
        assert "cat4_upstream_transport" in by_cat
        assert by_cat["cat1_purchased_goods"] == pytest.approx(100000.0 * 0.35, rel=0.01)
        assert by_cat["cat3_fuel_energy"] == pytest.approx(50000.0 * 0.55, rel=0.01)
        assert by_cat["cat4_upstream_transport"] == pytest.approx(30000.0 * 0.28, rel=0.01)


class TestPurchaseOrderSyncAndMatching:
    """Test PO sync and goods receipt matching."""

    def test_po_sync_and_full_match(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")
        pipeline.sync_purchase_orders(conn["connection_id"], SAMPLE_PURCHASE_ORDERS)

        # Full match: received exactly what was ordered
        result = pipeline.match_goods_receipt("PO-2025-0001", 700.0)
        assert result["status"] == "full_match"
        assert result["ordered_qty"] == 700.0
        assert result["received_qty"] == 700.0
        assert abs(result["variance_pct"]) < 1.0

    def test_po_partial_match(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")
        pipeline.sync_purchase_orders(conn["connection_id"], SAMPLE_PURCHASE_ORDERS)

        # Partial: received less than ordered
        result = pipeline.match_goods_receipt("PO-2025-0001", 500.0)
        assert result["status"] == "partial_match"
        assert result["variance_pct"] < 0

    def test_po_over_receipt(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")
        pipeline.sync_purchase_orders(conn["connection_id"], SAMPLE_PURCHASE_ORDERS)

        # Over receipt: received more than ordered
        result = pipeline.match_goods_receipt("PO-2025-0001", 800.0)
        assert result["status"] == "over_receipt"
        assert result["variance_pct"] > 0

    def test_po_not_found(self):
        pipeline = ERPConnectorPipeline()
        with pytest.raises(ValueError, match="Unknown PO"):
            pipeline.match_goods_receipt("PO-NONEXISTENT", 100.0)


class TestMultiCurrencyNormalization:
    """Test multi-currency spend normalization."""

    def test_normalize_all_records_to_usd(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")
        pipeline.sync_spend(conn["connection_id"], SAMPLE_SPEND_RECORDS_MULTI_CURRENCY)

        normalized = pipeline.normalize_spend_currency("USD")
        assert len(normalized) == 5

        # EUR record: 125000 * 1.08 = 135000 USD
        eur_record = next(r for r in normalized if r["record_id"] == "SPD-001")
        assert eur_record["amount_normalized"] == pytest.approx(135000.0, rel=0.01)

        # USD record: stays the same
        usd_record = next(r for r in normalized if r["record_id"] == "SPD-002")
        assert usd_record["amount_normalized"] == pytest.approx(78500.0, rel=0.01)

        # JPY record: 9500000 * 0.0067 = 63650 USD
        jpy_record = next(r for r in normalized if r["record_id"] == "SPD-003")
        assert jpy_record["amount_normalized"] == pytest.approx(63650.0, rel=0.01)

        # GBP record: 45000 * 1.27 = 57150 USD
        gbp_record = next(r for r in normalized if r["record_id"] == "SPD-004")
        assert gbp_record["amount_normalized"] == pytest.approx(57150.0, rel=0.01)

        # CNY record: 200000 * 0.14 = 28000 USD
        cny_record = next(r for r in normalized if r["record_id"] == "SPD-005")
        assert cny_record["amount_normalized"] == pytest.approx(28000.0, rel=0.01)

    def test_normalize_to_eur(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")
        pipeline.sync_spend(conn["connection_id"], [
            {"record_id": "S1", "amount": 1000.0, "currency": "USD"},
        ])
        normalized = pipeline.normalize_spend_currency("EUR")
        # 1000 USD * (1.0 / 1.08) = ~925.93 EUR
        assert normalized[0]["amount_normalized"] == pytest.approx(1000.0 / 1.08, rel=0.01)
        assert normalized[0]["normalized_currency"] == "EUR"


class TestVendorMappingOverrideChain:
    """Test vendor mapping override chain for Scope 3 classification."""

    def test_default_mapping_by_spend_category(self):
        pipeline = ERPConnectorPipeline()
        result = pipeline.classify_spend("raw_materials")
        assert result == "cat1_purchased_goods"

    def test_vendor_override_takes_precedence(self):
        pipeline = ERPConnectorPipeline()
        pipeline.map_vendor("V-SPECIAL", "Special Vendor", "cat6_business_travel")
        # Even though category is raw_materials, vendor override wins
        result = pipeline.classify_spend("raw_materials", vendor_id="V-SPECIAL")
        assert result == "cat6_business_travel"

    def test_no_vendor_mapping_falls_back_to_category(self):
        pipeline = ERPConnectorPipeline()
        pipeline.map_vendor("V-OTHER", "Other Vendor", "cat6_business_travel")
        # Different vendor, so falls back to category mapping
        result = pipeline.classify_spend("logistics", vendor_id="V-UNKNOWN")
        assert result == "cat4_upstream_transport"

    def test_unknown_category_returns_unclassified(self):
        pipeline = ERPConnectorPipeline()
        result = pipeline.classify_spend("unknown_category")
        assert result == "unclassified"


class TestProvenanceChainIntegrity:
    """Test provenance chain integrity across the full pipeline."""

    def test_full_pipeline_provenance(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("sap_s4hana", "sap.example.com")
        pipeline.test_connection(conn["connection_id"])
        pipeline.sync_spend(conn["connection_id"], [
            {"record_id": "S1", "amount": 10000.0, "currency": "USD",
             "category": "raw_materials"},
        ])

        chain = pipeline.get_provenance_chain(conn["connection_id"])
        assert len(chain) == 3
        assert chain[0]["operation"] == "register_connection"
        assert chain[1]["operation"] == "test_connection"
        assert chain[2]["operation"] == "sync_spend"

        # Verify chain links
        assert chain[0]["previous_hash"] == "0" * 64
        for i in range(1, len(chain)):
            assert chain[i]["previous_hash"] == chain[i - 1]["hash"]

    def test_provenance_verification_passes(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")
        pipeline.test_connection(conn["connection_id"])
        pipeline.sync_spend(conn["connection_id"], [
            {"record_id": "S1", "amount": 5000.0, "currency": "USD",
             "category": "fuel"},
        ])
        pipeline.sync_purchase_orders(conn["connection_id"], [
            {"po_number": "PO-001", "vendor_id": "V-001", "line_items": []},
        ])
        assert pipeline.verify_provenance(conn["connection_id"]) is True

    def test_empty_provenance_is_valid(self):
        pipeline = ERPConnectorPipeline()
        assert pipeline.verify_provenance("nonexistent") is True

    def test_provenance_timestamps_ordered(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")
        pipeline.test_connection(conn["connection_id"])
        pipeline.sync_spend(conn["connection_id"], [
            {"record_id": "S1", "amount": 1000.0, "currency": "USD",
             "category": "raw_materials"},
        ])
        chain = pipeline.get_provenance_chain(conn["connection_id"])
        timestamps = [r["timestamp"] for r in chain]
        assert timestamps == sorted(timestamps)


class TestErrorHandling:
    """Test error scenarios across the pipeline."""

    def test_sync_spend_unknown_connection(self):
        pipeline = ERPConnectorPipeline()
        with pytest.raises(ValueError, match="Unknown connection"):
            pipeline.sync_spend("conn-nonexistent", [])

    def test_sync_po_unknown_connection(self):
        pipeline = ERPConnectorPipeline()
        with pytest.raises(ValueError, match="Unknown connection"):
            pipeline.sync_purchase_orders("conn-nonexistent", [])

    def test_unsupported_currency_raises(self):
        pipeline = ERPConnectorPipeline()
        with pytest.raises(ValueError, match="Unsupported currency"):
            pipeline.convert_currency(1000.0, "ZZZ", "USD")

    def test_empty_spend_emissions_returns_zero(self):
        pipeline = ERPConnectorPipeline()
        result = pipeline.calculate_emissions([])
        assert result["total_emissions_kgco2e"] == 0.0
        assert result["records_calculated"] == 0


class TestStatisticsAfterPipeline:
    """Test statistics reflect pipeline state."""

    def test_statistics_after_full_pipeline(self):
        pipeline = ERPConnectorPipeline()
        conn = pipeline.register_connection("simulated", "localhost")
        pipeline.sync_spend(conn["connection_id"], SAMPLE_SPEND_RECORDS_MULTI_CURRENCY)
        pipeline.sync_purchase_orders(conn["connection_id"], SAMPLE_PURCHASE_ORDERS)
        pipeline.map_vendor("V-001", "EcoSteel GmbH", "cat1_purchased_goods")
        pipeline.calculate_emissions()

        stats = pipeline.get_statistics()
        assert stats["total_connections"] == 1
        assert stats["total_spend_records"] == 5
        assert stats["total_purchase_orders"] == 2
        assert stats["total_vendor_mappings"] == 1
        assert stats["total_emission_results"] == 5

    def test_initial_statistics_all_zero(self):
        pipeline = ERPConnectorPipeline()
        stats = pipeline.get_statistics()
        assert stats["total_connections"] == 0
        assert stats["total_spend_records"] == 0
        assert stats["total_purchase_orders"] == 0
        assert stats["total_vendor_mappings"] == 0
        assert stats["total_emission_results"] == 0
