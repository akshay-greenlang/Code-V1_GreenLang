# -*- coding: utf-8 -*-
"""
SectorClassificationBridge - NACE Sector Mapping for DMA PACK-015
===================================================================

This module provides NACE sector classification and mapping for the Double
Materiality Assessment. It maps company NACE codes to sector-specific
materiality profiles, provides sector benchmark data, adjusts default IRO
catalogs per sector, and supports multi-sector conglomerates with weighted
topic prioritization.

Features:
    - Map company NACE codes to sector-specific materiality profiles
    - Provide sector benchmark data for materiality scoring context
    - Adjust default IRO catalogs per sector
    - Support multi-sector conglomerates with revenue-weighted profiles
    - Pre-built profiles for 12 major NACE divisions
    - Cross-sector topic deduplication for conglomerates
    - SHA-256 provenance on all operations

NACE Divisions Covered:
    A  -- Agriculture, Forestry and Fishing
    B  -- Mining and Quarrying
    C  -- Manufacturing
    D  -- Electricity, Gas, Steam and Air Conditioning Supply
    F  -- Construction
    G  -- Wholesale and Retail Trade
    H  -- Transportation and Storage
    I  -- Accommodation and Food Service
    J  -- Information and Communication
    K  -- Financial and Insurance Activities
    L  -- Real Estate Activities
    Q  -- Human Health and Social Work Activities

Architecture:
    Company NACE Codes --> SectorClassificationBridge --> Sector Profile
                                    |                        |
                                    v                        v
    IRO Catalog Adjustment     Benchmark Data     Weighted Topic Priority
                                    |                        |
                                    v                        v
    DMA Engine Input <-- Provenance Hash <-- Conglomerate Merge

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-015 Double Materiality Assessment
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
# Data Models
# ---------------------------------------------------------------------------


class SectorBridgeConfig(BaseModel):
    """Configuration for the Sector Classification Bridge."""

    pack_id: str = Field(default="PACK-015")
    enable_provenance: bool = Field(default=True)
    enable_benchmark_data: bool = Field(default=True)
    conglomerate_weighting: str = Field(
        default="revenue",
        description="Weighting method for multi-sector: revenue, headcount, equal",
    )


class NACECode(BaseModel):
    """Structured NACE code with section and division info."""

    code: str = Field(..., description="Full NACE code (e.g., C25.1)")
    section: str = Field(default="", description="NACE section letter (e.g., C)")
    division: str = Field(default="", description="NACE division (e.g., 25)")
    description: str = Field(default="")
    revenue_share_pct: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Revenue share for conglomerate weighting",
    )


class TopicPriority(BaseModel):
    """Priority of an ESRS topic for a specific sector."""

    esrs_topic: str = Field(default="")
    topic_name: str = Field(default="")
    sector_priority: str = Field(
        default="medium",
        description="low, medium, high, critical",
    )
    default_material: bool = Field(
        default=False,
        description="Whether this topic is material by default for the sector",
    )
    benchmark_score: float = Field(
        default=0.0, ge=0.0, le=5.0,
        description="Typical materiality score in this sector",
    )


class SectorProfile(BaseModel):
    """Sector-specific materiality profile."""

    nace_section: str = Field(default="")
    sector_name: str = Field(default="")
    topic_priorities: List[TopicPriority] = Field(default_factory=list)
    default_material_topics: List[str] = Field(default_factory=list)
    iro_catalog_adjustments: Dict[str, str] = Field(
        default_factory=dict,
        description="IRO catalog modifications: topic -> adjustment type",
    )
    benchmark_data: Dict[str, Any] = Field(default_factory=dict)


class SectorClassificationResult(BaseModel):
    """Result of sector classification for a company."""

    result_id: str = Field(default_factory=_new_uuid)
    nace_codes: List[NACECode] = Field(default_factory=list)
    is_conglomerate: bool = Field(default=False)
    primary_sector: str = Field(default="")
    primary_sector_name: str = Field(default="")
    sector_profiles: List[SectorProfile] = Field(default_factory=list)
    merged_topic_priorities: List[TopicPriority] = Field(default_factory=list)
    default_material_topics: List[str] = Field(default_factory=list)
    iro_adjustments: Dict[str, str] = Field(default_factory=dict)
    success: bool = Field(default=False)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class SectorBenchmark(BaseModel):
    """Sector benchmark data for materiality comparison."""

    sector: str = Field(default="")
    sector_name: str = Field(default="")
    avg_material_topics: int = Field(default=0)
    common_material_topics: List[str] = Field(default_factory=list)
    avg_impact_score: float = Field(default=0.0, ge=0.0, le=5.0)
    avg_financial_score: float = Field(default=0.0, ge=0.0, le=5.0)
    peer_count: int = Field(default=0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Sector Profile Catalog (12 major NACE divisions)
# ---------------------------------------------------------------------------

SECTOR_PROFILES: Dict[str, SectorProfile] = {
    "A": SectorProfile(
        nace_section="A", sector_name="Agriculture, Forestry and Fishing",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="critical", default_material=True, benchmark_score=4.2),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="high", default_material=True, benchmark_score=3.8),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="critical", default_material=True, benchmark_score=4.0),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="critical", default_material=True, benchmark_score=4.5),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="medium", default_material=False, benchmark_score=2.8),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="high", default_material=True, benchmark_score=3.8),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="high", default_material=True, benchmark_score=3.6),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="medium", default_material=False, benchmark_score=2.9),
        ],
        default_material_topics=["E1", "E2", "E3", "E4", "S1", "S2", "S3"],
        iro_catalog_adjustments={"E4": "expand_biodiversity_iros", "E3": "expand_water_iros"},
        benchmark_data={"avg_material_topics": 7, "peer_count": 120},
    ),
    "B": SectorProfile(
        nace_section="B", sector_name="Mining and Quarrying",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="critical", default_material=True, benchmark_score=4.5),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="critical", default_material=True, benchmark_score=4.3),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="critical", default_material=True, benchmark_score=4.4),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="critical", default_material=True, benchmark_score=4.2),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="high", default_material=True, benchmark_score=3.8),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="critical", default_material=True, benchmark_score=4.0),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="critical", default_material=True, benchmark_score=4.3),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="low", default_material=False, benchmark_score=2.0),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="high", default_material=True, benchmark_score=3.7),
        ],
        default_material_topics=["E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "G1"],
        iro_catalog_adjustments={"S3": "expand_community_iros", "E2": "expand_pollution_iros"},
        benchmark_data={"avg_material_topics": 9, "peer_count": 85},
    ),
    "C": SectorProfile(
        nace_section="C", sector_name="Manufacturing",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="critical", default_material=True, benchmark_score=4.3),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="high", default_material=True, benchmark_score=3.6),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="medium", default_material=False, benchmark_score=2.9),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="high", default_material=True, benchmark_score=3.8),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="high", default_material=True, benchmark_score=3.4),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="medium", default_material=False, benchmark_score=2.7),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="high", default_material=True, benchmark_score=3.3),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="medium", default_material=False, benchmark_score=3.0),
        ],
        default_material_topics=["E1", "E2", "E5", "S1", "S2", "S4"],
        iro_catalog_adjustments={"E5": "expand_circular_iros"},
        benchmark_data={"avg_material_topics": 6, "peer_count": 350},
    ),
    "D": SectorProfile(
        nace_section="D", sector_name="Electricity, Gas, Steam and Air Conditioning Supply",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="critical", default_material=True, benchmark_score=4.8),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="high", default_material=True, benchmark_score=3.9),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="high", default_material=True, benchmark_score=3.4),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="medium", default_material=False, benchmark_score=2.8),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="medium", default_material=False, benchmark_score=2.6),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="high", default_material=True, benchmark_score=3.6),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="medium", default_material=False, benchmark_score=3.0),
        ],
        default_material_topics=["E1", "E2", "E3", "E4", "S1", "S3"],
        iro_catalog_adjustments={"E1": "expand_transition_iros"},
        benchmark_data={"avg_material_topics": 6, "peer_count": 95},
    ),
    "F": SectorProfile(
        nace_section="F", sector_name="Construction",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="high", default_material=True, benchmark_score=3.8),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="medium", default_material=False, benchmark_score=2.8),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="high", default_material=True, benchmark_score=3.6),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="critical", default_material=True, benchmark_score=4.2),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="high", default_material=True, benchmark_score=3.3),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="medium", default_material=False, benchmark_score=2.4),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="high", default_material=True, benchmark_score=3.4),
        ],
        default_material_topics=["E1", "E2", "E5", "S1", "S2", "S3", "G1"],
        iro_catalog_adjustments={"S1": "expand_health_safety_iros"},
        benchmark_data={"avg_material_topics": 7, "peer_count": 180},
    ),
    "G": SectorProfile(
        nace_section="G", sector_name="Wholesale and Retail Trade",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="low", default_material=False, benchmark_score=2.0),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="medium", default_material=False, benchmark_score=2.3),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="high", default_material=True, benchmark_score=3.4),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="high", default_material=True, benchmark_score=3.6),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="medium", default_material=False, benchmark_score=2.4),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="medium", default_material=False, benchmark_score=3.0),
        ],
        default_material_topics=["E1", "E5", "S1", "S2", "S4"],
        iro_catalog_adjustments={"S2": "expand_supply_chain_iros"},
        benchmark_data={"avg_material_topics": 5, "peer_count": 280},
    ),
    "H": SectorProfile(
        nace_section="H", sector_name="Transportation and Storage",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="critical", default_material=True, benchmark_score=4.6),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="high", default_material=True, benchmark_score=3.7),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="low", default_material=False, benchmark_score=1.8),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="medium", default_material=False, benchmark_score=2.6),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="high", default_material=True, benchmark_score=3.8),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="medium", default_material=False, benchmark_score=2.8),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="high", default_material=True, benchmark_score=3.2),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="medium", default_material=False, benchmark_score=2.9),
        ],
        default_material_topics=["E1", "E2", "S1", "S3"],
        iro_catalog_adjustments={"E1": "expand_transport_iros"},
        benchmark_data={"avg_material_topics": 4, "peer_count": 150},
    ),
    "I": SectorProfile(
        nace_section="I", sector_name="Accommodation and Food Service",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="high", default_material=True, benchmark_score=3.4),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="high", default_material=True, benchmark_score=3.3),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="medium", default_material=False, benchmark_score=2.6),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="high", default_material=True, benchmark_score=3.7),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="high", default_material=True, benchmark_score=3.4),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="medium", default_material=False, benchmark_score=2.8),
        ],
        default_material_topics=["E1", "E3", "E5", "S1", "S2", "S4"],
        iro_catalog_adjustments={"E5": "expand_food_waste_iros"},
        benchmark_data={"avg_material_topics": 6, "peer_count": 200},
    ),
    "J": SectorProfile(
        nace_section="J", sector_name="Information and Communication",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="high", default_material=True, benchmark_score=3.3),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="low", default_material=False, benchmark_score=1.8),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="low", default_material=False, benchmark_score=1.5),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="low", default_material=False, benchmark_score=1.5),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="high", default_material=True, benchmark_score=3.8),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="medium", default_material=False, benchmark_score=2.3),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="critical", default_material=True, benchmark_score=4.2),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="high", default_material=True, benchmark_score=3.8),
        ],
        default_material_topics=["E1", "S1", "S4", "G1"],
        iro_catalog_adjustments={"S4": "expand_data_privacy_iros", "G1": "expand_ethics_iros"},
        benchmark_data={"avg_material_topics": 4, "peer_count": 220},
    ),
    "K": SectorProfile(
        nace_section="K", sector_name="Financial and Insurance Activities",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="critical", default_material=True, benchmark_score=4.2),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="medium", default_material=False, benchmark_score=2.2),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="low", default_material=False, benchmark_score=1.8),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="low", default_material=False, benchmark_score=1.8),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="high", default_material=True, benchmark_score=3.5),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="medium", default_material=False, benchmark_score=2.3),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="high", default_material=True, benchmark_score=3.4),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="high", default_material=True, benchmark_score=3.6),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="critical", default_material=True, benchmark_score=4.5),
        ],
        default_material_topics=["E1", "S1", "S3", "S4", "G1"],
        iro_catalog_adjustments={"G1": "expand_financial_conduct_iros", "E1": "expand_financed_emissions_iros"},
        benchmark_data={"avg_material_topics": 5, "peer_count": 310},
    ),
    "L": SectorProfile(
        nace_section="L", sector_name="Real Estate Activities",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="critical", default_material=True, benchmark_score=4.3),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="medium", default_material=False, benchmark_score=2.6),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="high", default_material=True, benchmark_score=3.4),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="medium", default_material=False, benchmark_score=2.8),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="medium", default_material=False, benchmark_score=2.4),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="high", default_material=True, benchmark_score=3.3),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="medium", default_material=False, benchmark_score=2.9),
        ],
        default_material_topics=["E1", "E5", "S3"],
        iro_catalog_adjustments={"E1": "expand_building_energy_iros"},
        benchmark_data={"avg_material_topics": 3, "peer_count": 130},
    ),
    "Q": SectorProfile(
        nace_section="Q", sector_name="Human Health and Social Work Activities",
        topic_priorities=[
            TopicPriority(esrs_topic="E1", topic_name="Climate Change", sector_priority="medium", default_material=False, benchmark_score=2.8),
            TopicPriority(esrs_topic="E2", topic_name="Pollution", sector_priority="high", default_material=True, benchmark_score=3.3),
            TopicPriority(esrs_topic="E3", topic_name="Water and Marine Resources", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="E4", topic_name="Biodiversity and Ecosystems", sector_priority="low", default_material=False, benchmark_score=1.8),
            TopicPriority(esrs_topic="E5", topic_name="Resource Use and Circular Economy", sector_priority="high", default_material=True, benchmark_score=3.2),
            TopicPriority(esrs_topic="S1", topic_name="Own Workforce", sector_priority="critical", default_material=True, benchmark_score=4.5),
            TopicPriority(esrs_topic="S2", topic_name="Workers in the Value Chain", sector_priority="medium", default_material=False, benchmark_score=2.5),
            TopicPriority(esrs_topic="S3", topic_name="Affected Communities", sector_priority="high", default_material=True, benchmark_score=3.8),
            TopicPriority(esrs_topic="S4", topic_name="Consumers and End-Users", sector_priority="critical", default_material=True, benchmark_score=4.3),
            TopicPriority(esrs_topic="G1", topic_name="Business Conduct", sector_priority="high", default_material=True, benchmark_score=3.6),
        ],
        default_material_topics=["E2", "E5", "S1", "S3", "S4", "G1"],
        iro_catalog_adjustments={"S1": "expand_healthcare_worker_iros", "S4": "expand_patient_safety_iros"},
        benchmark_data={"avg_material_topics": 6, "peer_count": 110},
    ),
}


# ---------------------------------------------------------------------------
# SectorClassificationBridge
# ---------------------------------------------------------------------------


class SectorClassificationBridge:
    """NACE sector classification and materiality profile mapping for DMA.

    Maps company NACE codes to sector-specific materiality profiles, provides
    benchmark data, adjusts IRO catalogs, and supports multi-sector
    conglomerates with revenue-weighted topic prioritization.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = SectorClassificationBridge()
        >>> result = bridge.classify_company(["C25.1", "G47.19"])
        >>> print(f"Primary sector: {result.primary_sector_name}")
        >>> print(f"Default material topics: {result.default_material_topics}")
    """

    def __init__(self, config: Optional[SectorBridgeConfig] = None) -> None:
        """Initialize the Sector Classification Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or SectorBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("SectorClassificationBridge initialized")

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    def classify_company(
        self,
        nace_codes: List[str],
        revenue_shares: Optional[Dict[str, float]] = None,
    ) -> SectorClassificationResult:
        """Classify a company based on its NACE codes.

        Args:
            nace_codes: List of NACE codes (e.g., ['C25.1', 'G47.19']).
            revenue_shares: Optional dict mapping NACE codes to revenue %.

        Returns:
            SectorClassificationResult with profiles and topic priorities.
        """
        start = time.monotonic()

        parsed_codes = self._parse_nace_codes(nace_codes, revenue_shares)
        is_conglomerate = len(set(c.section for c in parsed_codes)) > 1

        # Get sector profiles
        profiles: List[SectorProfile] = []
        for code in parsed_codes:
            profile = SECTOR_PROFILES.get(code.section)
            if profile and profile not in profiles:
                profiles.append(profile)

        if not profiles:
            return SectorClassificationResult(
                nace_codes=parsed_codes,
                is_conglomerate=is_conglomerate,
                success=False,
                message=f"No sector profiles found for NACE codes: {nace_codes}",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # Determine primary sector (highest revenue share)
        primary = parsed_codes[0] if parsed_codes else None
        primary_section = primary.section if primary else ""
        primary_name = SECTOR_PROFILES.get(primary_section, SectorProfile()).sector_name

        # Merge topic priorities for conglomerates
        if is_conglomerate:
            merged_priorities = self._merge_priorities(parsed_codes, profiles)
        else:
            merged_priorities = profiles[0].topic_priorities if profiles else []

        # Aggregate default material topics
        default_topics = self._aggregate_default_topics(profiles)

        # Aggregate IRO adjustments
        iro_adjustments: Dict[str, str] = {}
        for profile in profiles:
            iro_adjustments.update(profile.iro_catalog_adjustments)

        elapsed = (time.monotonic() - start) * 1000

        result = SectorClassificationResult(
            nace_codes=parsed_codes,
            is_conglomerate=is_conglomerate,
            primary_sector=primary_section,
            primary_sector_name=primary_name,
            sector_profiles=profiles,
            merged_topic_priorities=merged_priorities,
            default_material_topics=default_topics,
            iro_adjustments=iro_adjustments,
            success=True,
            message=(
                f"Classified as {'conglomerate' if is_conglomerate else 'single-sector'}: "
                f"{primary_name}"
            ),
            duration_ms=elapsed,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Company classified: %d NACE codes, conglomerate=%s, primary=%s",
            len(nace_codes), is_conglomerate, primary_name,
        )
        return result

    # -------------------------------------------------------------------------
    # Benchmark Data
    # -------------------------------------------------------------------------

    def get_sector_benchmark(self, nace_section: str) -> Optional[SectorBenchmark]:
        """Get sector benchmark data for materiality comparison.

        Args:
            nace_section: NACE section letter (e.g., 'C', 'K').

        Returns:
            SectorBenchmark if found, None otherwise.
        """
        profile = SECTOR_PROFILES.get(nace_section)
        if profile is None:
            return None

        material_topics = profile.default_material_topics
        avg_impact = sum(
            t.benchmark_score for t in profile.topic_priorities if t.default_material
        )
        count = sum(1 for t in profile.topic_priorities if t.default_material) or 1

        benchmark = SectorBenchmark(
            sector=nace_section,
            sector_name=profile.sector_name,
            avg_material_topics=len(material_topics),
            common_material_topics=material_topics,
            avg_impact_score=round(avg_impact / count, 1),
            avg_financial_score=round(avg_impact / count * 0.9, 1),
            peer_count=profile.benchmark_data.get("peer_count", 0),
        )

        if self.config.enable_provenance:
            benchmark.provenance_hash = _compute_hash(benchmark)

        return benchmark

    def list_available_sectors(self) -> List[Dict[str, str]]:
        """List all available sector profiles.

        Returns:
            List of sector info dicts.
        """
        return [
            {"section": section, "name": profile.sector_name}
            for section, profile in SECTOR_PROFILES.items()
        ]

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _parse_nace_codes(
        self,
        codes: List[str],
        revenue_shares: Optional[Dict[str, float]] = None,
    ) -> List[NACECode]:
        """Parse raw NACE code strings into structured objects.

        Args:
            codes: List of NACE code strings.
            revenue_shares: Optional revenue share mapping.

        Returns:
            List of parsed NACECode objects.
        """
        parsed: List[NACECode] = []
        shares = revenue_shares or {}

        for code in codes:
            section = code[0] if code else ""
            parts = code[1:].split(".") if len(code) > 1 else [""]
            division = parts[0] if parts else ""

            profile = SECTOR_PROFILES.get(section)
            description = profile.sector_name if profile else f"Unknown section {section}"

            share = shares.get(code, 100.0 / len(codes))

            parsed.append(NACECode(
                code=code,
                section=section,
                division=division,
                description=description,
                revenue_share_pct=share,
            ))

        # Sort by revenue share descending
        parsed.sort(key=lambda c: c.revenue_share_pct, reverse=True)
        return parsed

    def _merge_priorities(
        self,
        codes: List[NACECode],
        profiles: List[SectorProfile],
    ) -> List[TopicPriority]:
        """Merge topic priorities from multiple sector profiles.

        Uses revenue-weighted averaging for benchmark scores.

        Args:
            codes: Parsed NACE codes with revenue shares.
            profiles: Sector profiles to merge.

        Returns:
            Merged list of TopicPriority objects.
        """
        section_weights: Dict[str, float] = {}
        for code in codes:
            if code.section not in section_weights:
                section_weights[code.section] = 0.0
            section_weights[code.section] += code.revenue_share_pct

        # Normalize weights to sum to 100
        total_weight = sum(section_weights.values()) or 1.0
        for section in section_weights:
            section_weights[section] = section_weights[section] / total_weight * 100.0

        topic_scores: Dict[str, Dict[str, Any]] = {}

        for profile in profiles:
            weight = section_weights.get(profile.nace_section, 0.0) / 100.0
            for tp in profile.topic_priorities:
                if tp.esrs_topic not in topic_scores:
                    topic_scores[tp.esrs_topic] = {
                        "topic_name": tp.topic_name,
                        "weighted_score": 0.0,
                        "max_priority": tp.sector_priority,
                        "any_default": tp.default_material,
                    }
                entry = topic_scores[tp.esrs_topic]
                entry["weighted_score"] += tp.benchmark_score * weight
                if tp.default_material:
                    entry["any_default"] = True

                priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                current = priority_order.get(entry["max_priority"], 0)
                incoming = priority_order.get(tp.sector_priority, 0)
                if incoming > current:
                    entry["max_priority"] = tp.sector_priority

        merged: List[TopicPriority] = []
        for topic, data in topic_scores.items():
            merged.append(TopicPriority(
                esrs_topic=topic,
                topic_name=data["topic_name"],
                sector_priority=data["max_priority"],
                default_material=data["any_default"],
                benchmark_score=round(data["weighted_score"], 1),
            ))

        merged.sort(key=lambda t: t.benchmark_score, reverse=True)
        return merged

    def _aggregate_default_topics(
        self,
        profiles: List[SectorProfile],
    ) -> List[str]:
        """Aggregate default material topics across profiles with deduplication.

        Args:
            profiles: Sector profiles.

        Returns:
            Deduplicated list of default material topic codes.
        """
        topics: List[str] = []
        seen: set = set()
        for profile in profiles:
            for topic in profile.default_material_topics:
                if topic not in seen:
                    topics.append(topic)
                    seen.add(topic)
        return topics
