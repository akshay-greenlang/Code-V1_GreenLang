# -*- coding: utf-8 -*-
"""
GreenLang Monitoring Integration Tests
======================================

Integration tests for the monitoring subsystem that require:
- Running Prometheus instance
- Running PushGateway instance
- Kubernetes cluster (for ServiceMonitor tests)
- S3 access (for Thanos tests)

These tests are typically run in CI/CD with real infrastructure
or using test containers.

Test Categories:
- test_prometheus_scrape: ServiceMonitor and target discovery
- test_thanos_upload: S3 block upload and store gateway queries
- test_alertmanager: Alert delivery and silence management
"""
