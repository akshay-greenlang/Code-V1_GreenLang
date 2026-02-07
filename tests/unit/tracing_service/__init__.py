# -*- coding: utf-8 -*-
"""
GreenLang Tracing Service Unit Tests (OBS-003)
===============================================

Unit tests for the OpenTelemetry distributed tracing SDK including:
- TracingConfig configuration and environment variable parsing
- TracerProvider setup, no-op fallback, and lifecycle management
- Auto-instrumentor setup for FastAPI, httpx, psycopg, redis, celery, requests
- Context propagation (inject/extract, W3C, baggage, tenant context)
- Decorators (@trace_operation, @trace_agent, @trace_pipeline)
- Span enrichment with GreenLang-specific GL_* attributes
- Custom GreenLangSampler with per-service-category overrides
- ASGI TracingMiddleware for request/response tracing
- MetricsBridge for Prometheus span-metric recording
- One-liner configure_tracing() setup entrypoint

These tests mock all OpenTelemetry dependencies so they can run without
the OTel SDK installed.
"""
