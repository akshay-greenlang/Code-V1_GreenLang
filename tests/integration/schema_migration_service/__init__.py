# -*- coding: utf-8 -*-
"""
Integration tests for the Schema Migration Agent Service (AGENT-DATA-017).

Tests multi-engine workflows across all seven engines:
  - SchemaRegistryEngine + SchemaVersionerEngine
  - ChangeDetectorEngine + CompatibilityCheckerEngine
  - Full pipeline: Registry -> Versioner -> Detector -> Checker -> Planner -> Executor

Author: GreenLang Platform Team
Date: February 2026
"""
