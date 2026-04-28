# -*- coding: utf-8 -*-
"""Phase 2 — URN compliance hardening test suite.

Per CTO Phase 2 brief Section 2.2 (URN compliance) — these tests
verify the URN parser/builder property contracts, the alias backfill
script's idempotency, the API/SDK primary-id audit fix-ups, and the
seed-data lowercase invariant.

The package marker is intentionally minimal — pytest discovers test
modules via filename, not package metadata. This file exists so the
``phase2`` directory is importable as a package by tools that walk
the test tree (e.g. coverage and IDE collectors).
"""
