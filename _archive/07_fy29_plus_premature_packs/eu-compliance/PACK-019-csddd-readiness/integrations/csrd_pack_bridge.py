# -*- coding: utf-8 -*-
"""
CSRDPackBridge - ESRS S1-S4/G1 to CSDDD Mapping Bridge for PACK-019
=======================================================================

This module maps ESRS social and governance disclosures (S1-S4, G1) to
corresponding CSDDD (Corporate Sustainability Due Diligence Directive)
requirements. It identifies overlapping disclosure obligations and gaps
between CSRD/ESRS reporting and CSDDD due diligence duties.

Legal References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D), Articles 5-12, 22
    - ESRS S1 (Own Workforce) -> CSDDD Art 5-10 (due diligence, own operations)
    - ESRS S2 (Value Chain Workers) -> CSDDD Art 6-10 (value chain due diligence)
    - ESRS S3 (Affected Communities) -> CSDDD Art 11-12 (stakeholder engagement)
    - ESRS S4 (Consumers and End-Users) -> CSDDD Art 6-7 (impact identification)
    - ESRS G1 (Business Conduct) -> CSDDD Art 5 (governance and policy integration)

ESRS Disclosure Requirement Mapping:
    S1: 17 DRs -> Art 5 (policy), Art 6 (identification), Art 8-9 (measures),
                  Art 10 (remediation), Art 13 (monitoring)
    S2: 5 DRs  -> Art 6 (identification), Art 7 (prioritisation),
                  Art 8 (prevention), Art 9 (cessation), Art 10 (remediation)
    S3: 5 DRs  -> Art 11 (stakeholder engagement), Art 12 (grievance)
    S4: 5 DRs  -> Art 6 (identification), Art 7 (prioritisation)
    G1: 6 DRs  -> Art 5 (policy integration and governance)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ESRSStandard(str, Enum):
    """ESRS social and governance standards relevant to CSDDD."""

    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    G1 = "G1"

class MappingCoverage(str, Enum):
    """Level of coverage between ESRS disclosure and CSDDD requirement."""

    FULL = "full"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    NONE = "none"

class GapSeverity(str, Enum):
    """Severity of identified gap between ESRS and CSDDD."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class BridgeConfig(BaseModel):
    """Configuration for the CSRD Pack Bridge."""

    pack_id: str = Field(default="PACK-019")
    source_pack_id: str = Field(default="PACK-017")
    enable_provenance: bool = Field(default=True)
    reporting_year: int = Field(default=2025, ge=2020, le=2030)

class ESRSDisclosure(BaseModel):
    """An ESRS disclosure requirement with its data."""

    dr_id: str = Field(default="", description="e.g. S1-1, S2-3")
    standard: ESRSStandard = Field(default=ESRSStandard.S1)
    title: str = Field(default="")
    is_reported: bool = Field(default=False)
    data: Dict[str, Any] = Field(default_factory=dict)

class CSDDDMapping(BaseModel):
    """Mapping between an ESRS DR and CSDDD article(s)."""

    dr_id: str = Field(default="")
    standard: ESRSStandard = Field(default=ESRSStandard.S1)
    csddd_articles: List[str] = Field(default_factory=list)
    coverage: MappingCoverage = Field(default=MappingCoverage.NONE)
    description: str = Field(default="")

class DisclosureGap(BaseModel):
    """Gap identified between ESRS disclosures and CSDDD requirements."""

    csddd_article: str = Field(default="")
    csddd_requirement: str = Field(default="")
    esrs_coverage: MappingCoverage = Field(default=MappingCoverage.NONE)
    severity: GapSeverity = Field(default=GapSeverity.MEDIUM)
    missing_disclosures: List[str] = Field(default_factory=list)
    recommendation: str = Field(default="")

class BridgeResult(BaseModel):
    """Result of a bridge mapping operation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    mappings: List[CSDDDMapping] = Field(default_factory=list)
    gaps: List[DisclosureGap] = Field(default_factory=list)
    coverage_summary: Dict[str, Any] = Field(default_factory=dict)
    records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# ESRS-to-CSDDD Mapping Tables
# ---------------------------------------------------------------------------

S1_MAPPING: Dict[str, Dict[str, Any]] = {
    "S1-1": {
        "title": "Policies related to own workforce",
        "csddd_articles": ["Art_5"],
        "coverage": MappingCoverage.FULL,
        "description": "Policy integration into corporate governance maps directly to Art 5",
    },
    "S1-2": {
        "title": "Processes for engaging with own workforce",
        "csddd_articles": ["Art_11"],
        "coverage": MappingCoverage.FULL,
        "description": "Stakeholder engagement with workers maps to Art 11",
    },
    "S1-3": {
        "title": "Processes to remediate negative impacts",
        "csddd_articles": ["Art_10"],
        "coverage": MappingCoverage.FULL,
        "description": "Remediation processes map directly to Art 10",
    },
    "S1-4": {
        "title": "Taking action on material impacts",
        "csddd_articles": ["Art_8", "Art_9"],
        "coverage": MappingCoverage.FULL,
        "description": "Prevention and cessation measures map to Art 8-9",
    },
    "S1-5": {
        "title": "Targets related to managing impacts",
        "csddd_articles": ["Art_13"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Target-setting partially maps to monitoring effectiveness",
    },
    "S1-6": {
        "title": "Characteristics of the undertaking's employees",
        "csddd_articles": ["Art_6"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Workforce data supports adverse impact identification",
    },
    "S1-7": {
        "title": "Characteristics of non-employee workers",
        "csddd_articles": ["Art_6"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Non-employee data supports impact identification",
    },
    "S1-8": {
        "title": "Collective bargaining coverage",
        "csddd_articles": ["Art_5", "Art_6"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Collective bargaining is a due diligence policy element",
    },
    "S1-9": {
        "title": "Diversity metrics",
        "csddd_articles": ["Art_6"],
        "coverage": MappingCoverage.MINIMAL,
        "description": "Diversity data supports non-discrimination impact assessment",
    },
    "S1-10": {
        "title": "Adequate wages",
        "csddd_articles": ["Art_6", "Art_8"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Adequate wage data is relevant to human rights due diligence",
    },
    "S1-11": {
        "title": "Social protection",
        "csddd_articles": ["Art_6"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Social protection data supports impact identification",
    },
    "S1-12": {
        "title": "Persons with disabilities",
        "csddd_articles": ["Art_6"],
        "coverage": MappingCoverage.MINIMAL,
        "description": "Disability data supports non-discrimination assessment",
    },
    "S1-13": {
        "title": "Training and skills development",
        "csddd_articles": ["Art_5"],
        "coverage": MappingCoverage.MINIMAL,
        "description": "Training policies may support due diligence capacity building",
    },
    "S1-14": {
        "title": "Health and safety metrics",
        "csddd_articles": ["Art_6", "Art_8"],
        "coverage": MappingCoverage.FULL,
        "description": "OHS data maps to human rights adverse impact identification",
    },
    "S1-15": {
        "title": "Work-life balance",
        "csddd_articles": ["Art_6"],
        "coverage": MappingCoverage.MINIMAL,
        "description": "Work-life balance data supports broader HR assessment",
    },
    "S1-16": {
        "title": "Remuneration metrics",
        "csddd_articles": ["Art_6"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Remuneration data relevant to living wage assessment",
    },
    "S1-17": {
        "title": "Incidents, complaints and severe human rights impacts",
        "csddd_articles": ["Art_10", "Art_12"],
        "coverage": MappingCoverage.FULL,
        "description": "Incident data maps directly to remediation and grievance",
    },
}

S2_MAPPING: Dict[str, Dict[str, Any]] = {
    "S2-1": {
        "title": "Policies related to value chain workers",
        "csddd_articles": ["Art_5", "Art_6"],
        "coverage": MappingCoverage.FULL,
        "description": "Value chain policies map to due diligence integration",
    },
    "S2-2": {
        "title": "Processes for engaging with value chain workers",
        "csddd_articles": ["Art_11"],
        "coverage": MappingCoverage.FULL,
        "description": "Stakeholder engagement in value chain maps to Art 11",
    },
    "S2-3": {
        "title": "Processes to remediate negative impacts on value chain workers",
        "csddd_articles": ["Art_9", "Art_10"],
        "coverage": MappingCoverage.FULL,
        "description": "Remediation for value chain workers maps to Art 9-10",
    },
    "S2-4": {
        "title": "Taking action on material impacts on value chain workers",
        "csddd_articles": ["Art_7", "Art_8"],
        "coverage": MappingCoverage.FULL,
        "description": "Prioritisation and prevention in value chain maps to Art 7-8",
    },
    "S2-5": {
        "title": "Targets related to managing impacts on value chain workers",
        "csddd_articles": ["Art_13"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Target-setting partially maps to monitoring effectiveness",
    },
}

S3_MAPPING: Dict[str, Dict[str, Any]] = {
    "S3-1": {
        "title": "Policies related to affected communities",
        "csddd_articles": ["Art_5", "Art_11"],
        "coverage": MappingCoverage.FULL,
        "description": "Community policies map to due diligence and engagement",
    },
    "S3-2": {
        "title": "Processes for engaging with affected communities",
        "csddd_articles": ["Art_11"],
        "coverage": MappingCoverage.FULL,
        "description": "Community engagement maps directly to Art 11",
    },
    "S3-3": {
        "title": "Processes to remediate negative impacts on communities",
        "csddd_articles": ["Art_10", "Art_12"],
        "coverage": MappingCoverage.FULL,
        "description": "Community remediation maps to Art 10-12 (grievance)",
    },
    "S3-4": {
        "title": "Taking action on material impacts on communities",
        "csddd_articles": ["Art_8", "Art_9"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Community actions partially covered by Art 8-9",
    },
    "S3-5": {
        "title": "Targets related to managing impacts on communities",
        "csddd_articles": ["Art_13"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Target-setting partially maps to monitoring effectiveness",
    },
}

S4_MAPPING: Dict[str, Dict[str, Any]] = {
    "S4-1": {
        "title": "Policies related to consumers and end-users",
        "csddd_articles": ["Art_5", "Art_6"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Consumer policies partially map to due diligence integration",
    },
    "S4-2": {
        "title": "Processes for engaging with consumers",
        "csddd_articles": ["Art_11"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Consumer engagement partially maps to stakeholder engagement",
    },
    "S4-3": {
        "title": "Processes to remediate negative impacts on consumers",
        "csddd_articles": ["Art_10"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Consumer remediation partially maps to Art 10",
    },
    "S4-4": {
        "title": "Taking action on impacts on consumers",
        "csddd_articles": ["Art_6", "Art_7"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Consumer actions map to impact identification and prioritisation",
    },
    "S4-5": {
        "title": "Targets related to managing consumer impacts",
        "csddd_articles": ["Art_13"],
        "coverage": MappingCoverage.MINIMAL,
        "description": "Target-setting minimally maps to monitoring effectiveness",
    },
}

G1_MAPPING: Dict[str, Dict[str, Any]] = {
    "G1-1": {
        "title": "Business conduct policies",
        "csddd_articles": ["Art_5"],
        "coverage": MappingCoverage.FULL,
        "description": "Business conduct policies map to due diligence policy integration",
    },
    "G1-2": {
        "title": "Management of relationships with suppliers",
        "csddd_articles": ["Art_5", "Art_16"],
        "coverage": MappingCoverage.FULL,
        "description": "Supplier management maps to policy integration and contractual clauses",
    },
    "G1-3": {
        "title": "Prevention and detection of corruption and bribery",
        "csddd_articles": ["Art_5", "Art_6"],
        "coverage": MappingCoverage.FULL,
        "description": "Anti-corruption maps to due diligence (Annex Part I, UNCAC)",
    },
    "G1-4": {
        "title": "Incidents of corruption or bribery",
        "csddd_articles": ["Art_6", "Art_10"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Corruption incidents partially map to impact ID and remediation",
    },
    "G1-5": {
        "title": "Political influence and lobbying activities",
        "csddd_articles": ["Art_5"],
        "coverage": MappingCoverage.MINIMAL,
        "description": "Lobbying disclosure has minimal direct CSDDD mapping",
    },
    "G1-6": {
        "title": "Payment practices",
        "csddd_articles": ["Art_5", "Art_8"],
        "coverage": MappingCoverage.PARTIAL,
        "description": "Payment practices relevant to supply chain due diligence",
    },
}

ALL_MAPPINGS: Dict[ESRSStandard, Dict[str, Dict[str, Any]]] = {
    ESRSStandard.S1: S1_MAPPING,
    ESRSStandard.S2: S2_MAPPING,
    ESRSStandard.S3: S3_MAPPING,
    ESRSStandard.S4: S4_MAPPING,
    ESRSStandard.G1: G1_MAPPING,
}

# CSDDD articles that must be covered
CSDDD_REQUIRED_ARTICLES: Dict[str, str] = {
    "Art_5": "Due diligence policy integration",
    "Art_6": "Identifying adverse impacts",
    "Art_7": "Prioritisation of impacts",
    "Art_8": "Preventing potential adverse impacts",
    "Art_9": "Bringing actual adverse impacts to an end",
    "Art_10": "Remediation",
    "Art_11": "Meaningful stakeholder engagement",
    "Art_12": "Grievance notification mechanism",
    "Art_13": "Monitoring effectiveness",
    "Art_16": "Model contractual clauses",
}

# ---------------------------------------------------------------------------
# CSRDPackBridge
# ---------------------------------------------------------------------------

class CSRDPackBridge:
    """ESRS S1-S4/G1 to CSDDD mapping bridge for PACK-019.

    Maps ESRS social and governance disclosures to CSDDD requirements,
    identifies coverage levels and gaps, and provides standard-specific
    mapping details for due diligence alignment.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = CSRDPackBridge(BridgeConfig())
        >>> esrs_data = {"S1": {"S1-1": {"is_reported": True}}}
        >>> result = bridge.map_esrs_to_csddd(esrs_data)
        >>> assert result.status == "completed"
    """

    def __init__(self, config: Optional[BridgeConfig] = None) -> None:
        """Initialize CSRDPackBridge."""
        self.config = config or BridgeConfig()
        logger.info(
            "CSRDPackBridge initialized (source=%s, target=%s)",
            self.config.source_pack_id,
            self.config.pack_id,
        )

    def map_esrs_to_csddd(self, esrs_data: Dict[str, Any]) -> BridgeResult:
        """Map ESRS S1-S4/G1 disclosures to CSDDD requirements.

        Args:
            esrs_data: Dict keyed by standard (S1, S2, S3, S4, G1) containing
                       disclosure data with DR IDs as sub-keys.

        Returns:
            BridgeResult with mappings, gaps, and coverage summary.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            all_mappings: List[CSDDDMapping] = []

            for standard in ESRSStandard:
                standard_data = esrs_data.get(standard.value, {})
                mapping_table = ALL_MAPPINGS.get(standard, {})

                for dr_id, mapping_info in mapping_table.items():
                    dr_data = standard_data.get(dr_id, {})
                    is_reported = dr_data.get("is_reported", False) if dr_data else False

                    coverage = (
                        MappingCoverage(mapping_info["coverage"])
                        if is_reported
                        else MappingCoverage.NONE
                    )

                    all_mappings.append(CSDDDMapping(
                        dr_id=dr_id,
                        standard=standard,
                        csddd_articles=mapping_info["csddd_articles"],
                        coverage=coverage,
                        description=mapping_info["description"],
                    ))

            result.mappings = all_mappings
            result.gaps = self.identify_disclosure_gaps(esrs_data)
            result.coverage_summary = self._compute_coverage_summary(all_mappings)
            result.records_processed = len(all_mappings)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)

            logger.info(
                "ESRS-to-CSDDD mapping: %d DRs mapped, %d gaps identified",
                len(all_mappings),
                len(result.gaps),
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("ESRS-to-CSDDD mapping failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_s1_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get S1 Own Workforce to CSDDD mapping table.

        Returns:
            Dict of S1 DR IDs to CSDDD mapping details.
        """
        return dict(S1_MAPPING)

    def get_s2_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get S2 Value Chain Workers to CSDDD mapping table.

        Returns:
            Dict of S2 DR IDs to CSDDD mapping details.
        """
        return dict(S2_MAPPING)

    def get_s3_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get S3 Affected Communities to CSDDD mapping table.

        Returns:
            Dict of S3 DR IDs to CSDDD mapping details.
        """
        return dict(S3_MAPPING)

    def get_s4_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get S4 Consumers and End-Users to CSDDD mapping table.

        Returns:
            Dict of S4 DR IDs to CSDDD mapping details.
        """
        return dict(S4_MAPPING)

    def get_g1_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get G1 Business Conduct to CSDDD mapping table.

        Returns:
            Dict of G1 DR IDs to CSDDD mapping details.
        """
        return dict(G1_MAPPING)

    def identify_disclosure_gaps(
        self,
        esrs_data: Dict[str, Any],
    ) -> List[DisclosureGap]:
        """Identify gaps between ESRS disclosures and CSDDD requirements.

        Args:
            esrs_data: Dict keyed by standard containing disclosure data.

        Returns:
            List of DisclosureGap objects for unmet CSDDD requirements.
        """
        # Build reverse map: CSDDD article -> covering DRs
        article_coverage: Dict[str, List[str]] = {
            art: [] for art in CSDDD_REQUIRED_ARTICLES
        }
        article_reported: Dict[str, List[str]] = {
            art: [] for art in CSDDD_REQUIRED_ARTICLES
        }

        for standard in ESRSStandard:
            standard_data = esrs_data.get(standard.value, {})
            mapping_table = ALL_MAPPINGS.get(standard, {})

            for dr_id, mapping_info in mapping_table.items():
                for article in mapping_info["csddd_articles"]:
                    if article in article_coverage:
                        article_coverage[article].append(dr_id)
                        dr_data = standard_data.get(dr_id, {})
                        if dr_data and dr_data.get("is_reported", False):
                            article_reported[article].append(dr_id)

        gaps: List[DisclosureGap] = []
        for article, description in CSDDD_REQUIRED_ARTICLES.items():
            covering_drs = article_coverage.get(article, [])
            reported_drs = article_reported.get(article, [])

            if not reported_drs:
                severity = GapSeverity.CRITICAL if covering_drs else GapSeverity.HIGH
                gaps.append(DisclosureGap(
                    csddd_article=article,
                    csddd_requirement=description,
                    esrs_coverage=MappingCoverage.NONE,
                    severity=severity,
                    missing_disclosures=covering_drs,
                    recommendation=(
                        f"Report ESRS disclosures {', '.join(covering_drs)} "
                        f"to satisfy {article}" if covering_drs
                        else f"No ESRS DR directly covers {article}; "
                             f"dedicated CSDDD compliance activity required"
                    ),
                ))
            elif len(reported_drs) < len(covering_drs):
                missing = [d for d in covering_drs if d not in reported_drs]
                gaps.append(DisclosureGap(
                    csddd_article=article,
                    csddd_requirement=description,
                    esrs_coverage=MappingCoverage.PARTIAL,
                    severity=GapSeverity.MEDIUM,
                    missing_disclosures=missing,
                    recommendation=(
                        f"Complete remaining disclosures {', '.join(missing)} "
                        f"for full {article} coverage"
                    ),
                ))

        logger.info("Identified %d disclosure gaps", len(gaps))
        return gaps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_coverage_summary(
        self,
        mappings: List[CSDDDMapping],
    ) -> Dict[str, Any]:
        """Compute coverage summary across all standards."""
        total = len(mappings)
        by_coverage = {
            MappingCoverage.FULL.value: 0,
            MappingCoverage.PARTIAL.value: 0,
            MappingCoverage.MINIMAL.value: 0,
            MappingCoverage.NONE.value: 0,
        }
        by_standard: Dict[str, Dict[str, int]] = {}

        for m in mappings:
            by_coverage[m.coverage.value] = by_coverage.get(m.coverage.value, 0) + 1

            std = m.standard.value
            if std not in by_standard:
                by_standard[std] = {"total": 0, "covered": 0}
            by_standard[std]["total"] += 1
            if m.coverage != MappingCoverage.NONE:
                by_standard[std]["covered"] += 1

        covered = total - by_coverage.get(MappingCoverage.NONE.value, 0)
        coverage_pct = round(covered / total * 100, 1) if total > 0 else 0.0

        return {
            "total_disclosures": total,
            "covered_disclosures": covered,
            "coverage_pct": coverage_pct,
            "by_coverage_level": by_coverage,
            "by_standard": by_standard,
        }
