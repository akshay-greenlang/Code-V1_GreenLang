# -*- coding: utf-8 -*-
"""
GreenLang Monitoring Unit Tests
===============================

Unit tests for the monitoring subsystem including:
- PushGateway client and BatchJobMetrics (OBS-001)
- Grafana SDK: GrafanaClient, DashboardBuilder, PanelBuilder, FolderManager (OBS-002)
- Prometheus metrics registration
- Recording rules validation
- Alert rules PromQL validation

These tests mock external dependencies (Prometheus, PushGateway, Grafana API)
to test the Python SDK in isolation.
"""
