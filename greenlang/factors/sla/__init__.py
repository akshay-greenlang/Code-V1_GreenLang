# -*- coding: utf-8 -*-
"""
Factors API Service Level Agreement (SLA) tracking (DEP9).

This package exposes the per-tenant SLA report surface:

  * :class:`SLATracker`  — compute uptime %, p95/p99 latency, error rate,
    and error-budget burn for a tenant over an arbitrary window.
  * FastAPI installer  :func:`install_sla_routes` — mounts
    ``/v1/sla/report`` and ``/v1/sla/burn`` on the Factors app.

Data sources:

  * Prometheus via :class:`greenlang.factors.sla.sla_tracker.PromQLClient`
    (real network) or the in-memory :class:`FakePromClient` used in tests.
  * Tenant metadata resolved via an injectable callback (defaults to the
    RBAC layer).

The separate ``greenlang/factors/ga/sla_tracker.py`` covers *service-wide*
SLA compliance for GA go/no-go (F102). This module is the **per-tenant
contractual SLA report** — the one we hand to Enterprise customers on a
monthly cadence or embed in invoices as uptime credits.
"""
from greenlang.factors.sla.sla_tracker import (
    BurnRate,
    FakePromClient,
    PromQLClient,
    SLAReport,
    SLATracker,
    install_sla_routes,
)

__all__ = [
    "BurnRate",
    "FakePromClient",
    "PromQLClient",
    "SLAReport",
    "SLATracker",
    "install_sla_routes",
]
