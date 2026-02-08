# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-003: ERP/Finance Connector

12 Prometheus metrics for ERP connector service monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_erp_connections_total (Counter, labels: erp_system, tenant_id)
    2.  gl_erp_sync_duration_seconds (Histogram, 12 buckets)
    3.  gl_erp_spend_records_total (Counter, labels: spend_category)
    4.  gl_erp_purchase_orders_total (Counter, labels: status)
    5.  gl_erp_scope3_mappings_total (Counter, labels: category)
    6.  gl_erp_emissions_calculated_total (Counter, labels: methodology)
    7.  gl_erp_sync_errors_total (Counter, labels: erp_system, error_type)
    8.  gl_erp_currency_conversions_total (Counter, labels: from_currency, to_currency)
    9.  gl_erp_inventory_items_total (Counter, labels: material_group)
    10. gl_erp_batch_syncs_total (Counter, labels: status)
    11. gl_erp_active_connections (Gauge)
    12. gl_erp_sync_queue_size (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; ERP connector metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. ERP connections established by system type and tenant
    erp_connections_total = Counter(
        "gl_erp_connections_total",
        "Total ERP connections registered",
        labelnames=["erp_system", "tenant_id"],
    )

    # 2. Sync duration histogram (12 buckets covering sub-second to
    #    multi-minute long-running ERP syncs)
    erp_sync_duration_seconds = Histogram(
        "gl_erp_sync_duration_seconds",
        "ERP sync operation duration in seconds",
        buckets=(
            0.1, 0.5, 1.0, 2.5, 5.0, 10.0,
            30.0, 60.0, 120.0, 300.0, 480.0, 600.0,
        ),
    )

    # 3. Spend records ingested by spend category
    erp_spend_records_total = Counter(
        "gl_erp_spend_records_total",
        "Total spend records synced from ERP",
        labelnames=["spend_category"],
    )

    # 4. Purchase orders synced by status
    erp_purchase_orders_total = Counter(
        "gl_erp_purchase_orders_total",
        "Total purchase orders synced from ERP",
        labelnames=["status"],
    )

    # 5. Scope 3 category mappings created
    erp_scope3_mappings_total = Counter(
        "gl_erp_scope3_mappings_total",
        "Total Scope 3 category mappings applied",
        labelnames=["category"],
    )

    # 6. Emissions calculations performed by methodology
    erp_emissions_calculated_total = Counter(
        "gl_erp_emissions_calculated_total",
        "Total emissions calculations performed",
        labelnames=["methodology"],
    )

    # 7. Sync errors by ERP system and error type
    erp_sync_errors_total = Counter(
        "gl_erp_sync_errors_total",
        "Total sync errors encountered",
        labelnames=["erp_system", "error_type"],
    )

    # 8. Currency conversions performed
    erp_currency_conversions_total = Counter(
        "gl_erp_currency_conversions_total",
        "Total currency conversions performed",
        labelnames=["from_currency", "to_currency"],
    )

    # 9. Inventory items synced by material group
    erp_inventory_items_total = Counter(
        "gl_erp_inventory_items_total",
        "Total inventory items synced from ERP",
        labelnames=["material_group"],
    )

    # 10. Batch sync operations by status
    erp_batch_syncs_total = Counter(
        "gl_erp_batch_syncs_total",
        "Total batch sync operations",
        labelnames=["status"],
    )

    # 11. Currently active ERP connections
    erp_active_connections = Gauge(
        "gl_erp_active_connections",
        "Number of currently active ERP connections",
    )

    # 12. Current sync queue depth
    erp_sync_queue_size = Gauge(
        "gl_erp_sync_queue_size",
        "Current number of sync operations waiting in queue",
    )

else:
    # No-op placeholders
    erp_connections_total = None  # type: ignore[assignment]
    erp_sync_duration_seconds = None  # type: ignore[assignment]
    erp_spend_records_total = None  # type: ignore[assignment]
    erp_purchase_orders_total = None  # type: ignore[assignment]
    erp_scope3_mappings_total = None  # type: ignore[assignment]
    erp_emissions_calculated_total = None  # type: ignore[assignment]
    erp_sync_errors_total = None  # type: ignore[assignment]
    erp_currency_conversions_total = None  # type: ignore[assignment]
    erp_inventory_items_total = None  # type: ignore[assignment]
    erp_batch_syncs_total = None  # type: ignore[assignment]
    erp_active_connections = None  # type: ignore[assignment]
    erp_sync_queue_size = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_connection(
    erp_system: str,
    tenant_id: str,
) -> None:
    """Record an ERP connection registration.

    Args:
        erp_system: ERP system type (sap, oracle, netsuite, dynamics, simulated).
        tenant_id: Tenant identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    erp_connections_total.labels(
        erp_system=erp_system, tenant_id=tenant_id,
    ).inc()


def record_spend_record(
    spend_category: str,
    count: int = 1,
) -> None:
    """Record spend records synced from ERP.

    Args:
        spend_category: Spend category classification.
        count: Number of records to increment by.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    erp_spend_records_total.labels(
        spend_category=spend_category,
    ).inc(count)


def record_purchase_order(status: str) -> None:
    """Record a purchase order sync event.

    Args:
        status: Purchase order status (open, closed, cancelled, pending).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    erp_purchase_orders_total.labels(status=status).inc()


def record_scope3_mapping(category: str) -> None:
    """Record a Scope 3 category mapping applied.

    Args:
        category: Scope 3 category (e.g. cat1_purchased_goods, cat6_business_travel).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    erp_scope3_mappings_total.labels(category=category).inc()


def record_emissions_calculated(methodology: str) -> None:
    """Record an emissions calculation event.

    Args:
        methodology: Methodology used (eeio, spend_based, hybrid, supplier_specific).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    erp_emissions_calculated_total.labels(methodology=methodology).inc()


def record_sync_error(erp_system: str, error_type: str) -> None:
    """Record a sync error event.

    Args:
        erp_system: ERP system that produced the error.
        error_type: Error classification (connection, timeout, auth, data, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    erp_sync_errors_total.labels(
        erp_system=erp_system, error_type=error_type,
    ).inc()


def record_currency_conversion(
    from_currency: str,
    to_currency: str,
) -> None:
    """Record a currency conversion operation.

    Args:
        from_currency: Source currency code (e.g. EUR).
        to_currency: Target currency code (e.g. USD).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    erp_currency_conversions_total.labels(
        from_currency=from_currency, to_currency=to_currency,
    ).inc()


def record_inventory_item(material_group: str, count: int = 1) -> None:
    """Record inventory items synced from ERP.

    Args:
        material_group: Material group classification.
        count: Number of items to increment by.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    erp_inventory_items_total.labels(
        material_group=material_group,
    ).inc(count)


def record_batch_sync(status: str) -> None:
    """Record a batch sync operation event.

    Args:
        status: Batch sync status (submitted, completed, failed, partial).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    erp_batch_syncs_total.labels(status=status).inc()


def update_active_connections(delta: int) -> None:
    """Update the active ERP connections gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    if delta > 0:
        erp_active_connections.inc(delta)
    elif delta < 0:
        erp_active_connections.dec(abs(delta))


def update_sync_queue_size(size: int) -> None:
    """Set the current sync queue size.

    Args:
        size: Current queue depth.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    erp_sync_queue_size.set(size)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "erp_connections_total",
    "erp_sync_duration_seconds",
    "erp_spend_records_total",
    "erp_purchase_orders_total",
    "erp_scope3_mappings_total",
    "erp_emissions_calculated_total",
    "erp_sync_errors_total",
    "erp_currency_conversions_total",
    "erp_inventory_items_total",
    "erp_batch_syncs_total",
    "erp_active_connections",
    "erp_sync_queue_size",
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
