# -*- coding: utf-8 -*-
"""
CTO ten logical services mapped to in-repo packages (X-ten-modules).

Single deployable today; split to workers later without renaming these anchors.
"""

from __future__ import annotations

from typing import Dict

SERVICE_MODULE_MAP: Dict[str, str] = {
    "1_source_registry": "greenlang.factors.source_registry",
    "1g_governance_gates": "greenlang.factors.approval_gate",
    "2_ingestion_artifact_store": "greenlang.factors.ingestion.artifacts",
    "3_parser_service": "greenlang.factors.ingestion.parser_harness",
    "4_canonical_normalizer": "greenlang.factors.ingestion.normalizer",
    "5_quality_engine": "greenlang.factors.quality.validators",
    "6_policy_method_store": "greenlang.factors.policy_mapping",
    "7_search_matching": "greenlang.factors.matching.pipeline",
    "8_release_manager": "greenlang.factors.edition_manifest",
    "9_api_gateway": "greenlang.integration.api.main",
    "10_admin_console": "greenlang.factors.cli",
}

__all__ = ["SERVICE_MODULE_MAP"]
