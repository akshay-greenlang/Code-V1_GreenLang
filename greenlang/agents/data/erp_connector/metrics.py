# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-003: ERP/Finance Connector

12 Prometheus metrics for ERP connector service monitoring with graceful
fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_erp_operations_total (Counter, labels: type, tenant_id)
    2.  gl_erp_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_erp_validation_errors_total (Counter, labels: severity, type)
    4.  gl_erp_batch_jobs_total (Counter, labels: status)
    5.  gl_erp_active_jobs (Gauge)
    6.  gl_erp_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_erp_spend_records_total (Counter, labels: spend_category)
    8.  gl_erp_purchase_orders_total (Counter, labels: status)
    9.  gl_erp_scope3_mappings_total (Counter, labels: category)
    10. gl_erp_emissions_calculated_total (Counter, labels: methodology)
    11. gl_erp_currency_conversions_total (Counter, labels: from_currency, to_currency)
    12. gl_erp_inventory_items_total (Counter, labels: material_group)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    LONG_DURATION_BUCKETS,
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory(
    "gl_erp",
    "ERP Connector",
    duration_buckets=LONG_DURATION_BUCKETS,
)

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

erp_spend_records_total = m.create_custom_counter(
    "spend_records_total",
    "Total spend records synced from ERP",
    labelnames=["spend_category"],
)

erp_purchase_orders_total = m.create_custom_counter(
    "purchase_orders_total",
    "Total purchase orders synced from ERP",
    labelnames=["status"],
)

erp_scope3_mappings_total = m.create_custom_counter(
    "scope3_mappings_total",
    "Total Scope 3 category mappings applied",
    labelnames=["category"],
)

erp_emissions_calculated_total = m.create_custom_counter(
    "emissions_calculated_total",
    "Total emissions calculations performed",
    labelnames=["methodology"],
)

erp_currency_conversions_total = m.create_custom_counter(
    "currency_conversions_total",
    "Total currency conversions performed",
    labelnames=["from_currency", "to_currency"],
)

erp_inventory_items_total = m.create_custom_counter(
    "inventory_items_total",
    "Total inventory items synced from ERP",
    labelnames=["material_group"],
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_connection(erp_system: str, tenant_id: str) -> None:
    """Record an ERP connection registration.

    Args:
        erp_system: ERP system type (sap, oracle, netsuite, dynamics, simulated).
        tenant_id: Tenant identifier.
    """
    m.record_operation(0.0, type=erp_system, tenant_id=tenant_id)


def record_spend_record(spend_category: str, count: int = 1) -> None:
    """Record spend records synced from ERP.

    Args:
        spend_category: Spend category classification.
        count: Number of records to increment by.
    """
    m.safe_inc(erp_spend_records_total, count, spend_category=spend_category)


def record_purchase_order(status: str) -> None:
    """Record a purchase order sync event.

    Args:
        status: Purchase order status (open, closed, cancelled, pending).
    """
    m.safe_inc(erp_purchase_orders_total, 1, status=status)


def record_scope3_mapping(category: str) -> None:
    """Record a Scope 3 category mapping applied.

    Args:
        category: Scope 3 category (e.g. cat1_purchased_goods, cat6_business_travel).
    """
    m.safe_inc(erp_scope3_mappings_total, 1, category=category)


def record_emissions_calculated(methodology: str) -> None:
    """Record an emissions calculation event.

    Args:
        methodology: Methodology used (eeio, spend_based, hybrid, supplier_specific).
    """
    m.safe_inc(erp_emissions_calculated_total, 1, methodology=methodology)


def record_sync_error(erp_system: str, error_type: str) -> None:
    """Record a sync error event.

    Args:
        erp_system: ERP system that produced the error.
        error_type: Error classification (connection, timeout, auth, data, unknown).
    """
    m.record_validation_error(severity=erp_system, type=error_type)


def record_currency_conversion(from_currency: str, to_currency: str) -> None:
    """Record a currency conversion operation.

    Args:
        from_currency: Source currency code (e.g. EUR).
        to_currency: Target currency code (e.g. USD).
    """
    m.safe_inc(
        erp_currency_conversions_total, 1,
        from_currency=from_currency, to_currency=to_currency,
    )


def record_inventory_item(material_group: str, count: int = 1) -> None:
    """Record inventory items synced from ERP.

    Args:
        material_group: Material group classification.
        count: Number of items to increment by.
    """
    m.safe_inc(erp_inventory_items_total, count, material_group=material_group)


def record_batch_sync(status: str) -> None:
    """Record a batch sync operation event.

    Args:
        status: Batch sync status (submitted, completed, failed, partial).
    """
    m.record_batch_job(status)


def update_active_connections(delta: int) -> None:
    """Update the active ERP connections gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    m.update_active_jobs(delta)


def update_sync_queue_size(size: int) -> None:
    """Set the current sync queue size.

    Args:
        size: Current queue depth.
    """
    m.update_queue_size(size)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Agent-specific metric objects
    "erp_spend_records_total",
    "erp_purchase_orders_total",
    "erp_scope3_mappings_total",
    "erp_emissions_calculated_total",
    "erp_currency_conversions_total",
    "erp_inventory_items_total",
    # Helper functions
    "record_connection",
    "record_spend_record",
    "record_purchase_order",
    "record_scope3_mapping",
    "record_emissions_calculated",
    "record_sync_error",
    "record_currency_conversion",
    "record_inventory_item",
    "record_batch_sync",
    "update_active_connections",
    "update_sync_queue_size",
]
