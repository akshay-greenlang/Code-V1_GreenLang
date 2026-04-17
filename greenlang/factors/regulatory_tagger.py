# -*- coding: utf-8 -*-
"""
Regulatory tagger for emission factors (F083).

Automatically tags factors with the regulatory frameworks they satisfy
(GHG Protocol, ISO 14064, CSRD/ESRS, CBAM, SBTi, CDP, EUDR, Taxonomy).
Enables framework-filtered search and compliance gap analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RegulatoryFramework(str, Enum):
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CBAM = "cbam"
    SBTI = "sbti"
    CDP = "cdp"
    EUDR = "eudr"
    EU_TAXONOMY = "eu_taxonomy"
    EPA_GHGRP = "epa_ghgrp"
    UK_SECR = "uk_secr"


@dataclass
class TaggingRule:
    """Rule that maps factor attributes to a regulatory framework."""

    framework: RegulatoryFramework
    description: str
    match_categories: Set[str] = field(default_factory=set)
    match_sources: Set[str] = field(default_factory=set)
    match_geographies: Set[str] = field(default_factory=set)
    match_scopes: Set[str] = field(default_factory=set)
    requires_certified: bool = False
    min_dqs: float = 0.0

    def matches(self, factor: Dict[str, Any]) -> bool:
        """Check if a factor matches this rule."""
        if self.match_categories:
            cat = factor.get("category", "")
            if not any(c in cat for c in self.match_categories):
                return False
        if self.match_sources:
            src = factor.get("source_id", "")
            if src not in self.match_sources:
                return False
        if self.match_geographies:
            geo = factor.get("geography", "")
            if geo not in self.match_geographies and "GLOBAL" not in self.match_geographies:
                return False
        if self.match_scopes:
            scope = str(factor.get("scope", ""))
            if scope not in self.match_scopes:
                return False
        if self.requires_certified:
            if factor.get("status") != "certified":
                return False
        if self.min_dqs > 0:
            dqs = factor.get("data_quality_score", 0)
            if dqs < self.min_dqs:
                return False
        return True


# Default regulatory tagging rules
DEFAULT_RULES: List[TaggingRule] = [
    TaggingRule(
        framework=RegulatoryFramework.GHG_PROTOCOL,
        description="GHG Protocol Corporate Standard — Scope 1-3 emission factors",
        match_scopes={"1", "2", "3"},
    ),
    TaggingRule(
        framework=RegulatoryFramework.ISO_14064,
        description="ISO 14064 — all certified factors with DQS >= 3",
        requires_certified=True,
        min_dqs=3.0,
    ),
    TaggingRule(
        framework=RegulatoryFramework.CSRD_ESRS,
        description="CSRD/ESRS E1 — climate-related emission factors",
        match_categories={"energy", "transport", "industrial", "waste", "agriculture"},
    ),
    TaggingRule(
        framework=RegulatoryFramework.CBAM,
        description="CBAM — factors for iron, steel, aluminium, cement, fertilizers, electricity, hydrogen",
        match_categories={"cement", "steel", "iron", "aluminium", "fertilizer", "electricity", "hydrogen"},
    ),
    TaggingRule(
        framework=RegulatoryFramework.SBTI,
        description="SBTi — science-based target factors (Scope 1-2 + relevant Scope 3)",
        match_scopes={"1", "2", "3"},
        min_dqs=2.0,
    ),
    TaggingRule(
        framework=RegulatoryFramework.CDP,
        description="CDP — disclosure-quality emission factors",
        match_scopes={"1", "2", "3"},
    ),
    TaggingRule(
        framework=RegulatoryFramework.EUDR,
        description="EUDR — deforestation-linked commodity factors",
        match_categories={"palm_oil", "soy", "cocoa", "coffee", "rubber", "cattle", "wood"},
    ),
    TaggingRule(
        framework=RegulatoryFramework.EU_TAXONOMY,
        description="EU Taxonomy — climate mitigation/adaptation aligned factors",
        match_categories={"energy", "transport", "manufacturing", "construction", "forestry"},
    ),
    TaggingRule(
        framework=RegulatoryFramework.EPA_GHGRP,
        description="EPA GHGRP — US-geography factors from EPA sources",
        match_sources={"epa_ghg", "epa_egrid"},
        match_geographies={"US", "USA"},
    ),
    TaggingRule(
        framework=RegulatoryFramework.UK_SECR,
        description="UK SECR — DEFRA/BEIS factors for UK reporting",
        match_sources={"defra_2025", "defra_2024", "defra_2023"},
        match_geographies={"GB", "UK"},
    ),
]


@dataclass
class TagResult:
    """Result of tagging a single factor."""

    factor_id: str
    frameworks: List[RegulatoryFramework]
    rule_matches: Dict[str, List[str]] = field(default_factory=dict)


class RegulatoryTagger:
    """
    Tags emission factors with applicable regulatory frameworks.

    Usage:
        tagger = RegulatoryTagger()
        result = tagger.tag_factor(factor_dict)
        # result.frameworks -> [GHG_PROTOCOL, CSRD_ESRS, ...]
    """

    def __init__(self, rules: Optional[List[TaggingRule]] = None) -> None:
        self._rules = rules or list(DEFAULT_RULES)

    def add_rule(self, rule: TaggingRule) -> None:
        """Add a custom tagging rule."""
        self._rules.append(rule)

    def tag_factor(self, factor: Dict[str, Any]) -> TagResult:
        """Tag a single factor with all matching frameworks."""
        factor_id = factor.get("factor_id", "unknown")
        matched: List[RegulatoryFramework] = []
        rule_matches: Dict[str, List[str]] = {}

        for rule in self._rules:
            if rule.matches(factor):
                matched.append(rule.framework)
                rule_matches.setdefault(rule.framework.value, []).append(rule.description)

        return TagResult(
            factor_id=factor_id,
            frameworks=matched,
            rule_matches=rule_matches,
        )

    def tag_batch(self, factors: List[Dict[str, Any]]) -> List[TagResult]:
        """Tag a batch of factors."""
        return [self.tag_factor(f) for f in factors]

    def coverage_report(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a coverage report showing how many factors satisfy each framework.
        """
        results = self.tag_batch(factors)
        coverage: Dict[str, int] = {fw.value: 0 for fw in RegulatoryFramework}
        for r in results:
            for fw in r.frameworks:
                coverage[fw.value] += 1

        total = len(factors)
        return {
            "total_factors": total,
            "framework_coverage": {
                fw: {"count": count, "ratio": round(count / total, 4) if total else 0.0}
                for fw, count in coverage.items()
            },
            "untagged_count": sum(1 for r in results if not r.frameworks),
        }

    def filter_by_framework(
        self, factors: List[Dict[str, Any]], framework: RegulatoryFramework
    ) -> List[Dict[str, Any]]:
        """Return only factors matching a specific regulatory framework."""
        return [
            f for f in factors
            if any(r.framework == framework and r.matches(f) for r in self._rules)
        ]

    @property
    def rules(self) -> List[TaggingRule]:
        return list(self._rules)
