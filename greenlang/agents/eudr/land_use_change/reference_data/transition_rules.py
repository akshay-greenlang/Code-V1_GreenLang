# -*- coding: utf-8 -*-
"""
Regulatory Transition Classification Rules - AGENT-EUDR-005

Provides comprehensive rules for classifying land use transitions under EUDR
regulation. Includes:
    - Full 10x10 transition classification matrix
    - Severity classification with EUDR article references
    - Verdict determination rules for CutoffDateVerifier
    - Commodity-specific conversion rules

All rules are deterministic and derived directly from EUDR Regulation
(EU) 2023/1115 articles and IPCC AFOLU 2006 Guidelines.

EUDR Articles Referenced:
    Article 2(1)  - Deforestation definition
    Article 2(4)  - Plantation forest exclusion
    Article 2(5)  - Forest degradation definition
    Article 3(a)  - Prohibition on deforestation-linked products
    Article 3(b)  - Legality requirement
    Article 9(1)  - Due diligence statement requirements
    Article 10(2) - Risk assessment criteria

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent
Status: Production Ready
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.agents.eudr.land_use_change.reference_data.land_use_parameters import (
    LandUseCategory,
)


# ---------------------------------------------------------------------------
# Transition type enumeration
# ---------------------------------------------------------------------------


class TransitionType(str, enum.Enum):
    """Classification of land use transition types.

    Maps each possible (from, to) land use transition to a regulatory
    category aligned with EUDR and IPCC terminology.
    """

    DEFORESTATION = "deforestation"
    DEGRADATION = "degradation"
    REFORESTATION = "reforestation"
    AFFORESTATION = "afforestation"
    AGRICULTURAL_EXPANSION = "agricultural_expansion"
    AGRICULTURAL_CONVERSION = "agricultural_conversion"
    AGRICULTURAL_ABANDONMENT = "agricultural_abandonment"
    URBANIZATION = "urbanization"
    WETLAND_CONVERSION = "wetland_conversion"
    WETLAND_RESTORATION = "wetland_restoration"
    COMMODITY_CONVERSION = "commodity_conversion"
    PLANTATION_MANAGEMENT = "plantation_management"
    STABLE = "stable"
    NATURAL_SUCCESSION = "natural_succession"
    INUNDATION = "inundation"
    LAND_RECLAMATION = "land_reclamation"
    UNCLASSIFIED = "unclassified"


# ---------------------------------------------------------------------------
# Compliance verdict enumeration
# ---------------------------------------------------------------------------


class ComplianceVerdict(str, enum.Enum):
    """EUDR compliance verdict for a land use transition assessment.

    Used by CutoffDateVerifier to assign a final compliance determination.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    INSUFFICIENT_DATA = "insufficient_data"
    EXCLUDED = "excluded"


# ---------------------------------------------------------------------------
# Transition classification rules (10 x 10 = 100 combinations)
# ---------------------------------------------------------------------------
#
# Maps every (from_class, to_class) pair to its TransitionType.
# Diagonal entries (same class) are STABLE.
# All 100 combinations are explicitly defined to avoid fallback ambiguity.

_FL = LandUseCategory.FOREST_LAND
_CL = LandUseCategory.CROPLAND
_GL = LandUseCategory.GRASSLAND
_WL = LandUseCategory.WETLAND
_SL = LandUseCategory.SETTLEMENT
_OL = LandUseCategory.OTHER_LAND
_OP = LandUseCategory.OIL_PALM_PLANTATION
_RP = LandUseCategory.RUBBER_PLANTATION
_PF = LandUseCategory.PLANTATION_FOREST
_WB = LandUseCategory.WATER_BODY

TRANSITION_CLASSIFICATION_RULES: Dict[
    Tuple[LandUseCategory, LandUseCategory], TransitionType
] = {
    # ---- FROM: FOREST_LAND ----
    (_FL, _FL): TransitionType.STABLE,
    (_FL, _CL): TransitionType.DEFORESTATION,
    (_FL, _GL): TransitionType.DEFORESTATION,
    (_FL, _WL): TransitionType.DEFORESTATION,
    (_FL, _SL): TransitionType.DEFORESTATION,
    (_FL, _OL): TransitionType.DEFORESTATION,
    (_FL, _OP): TransitionType.DEFORESTATION,
    (_FL, _RP): TransitionType.DEFORESTATION,
    (_FL, _PF): TransitionType.DEGRADATION,
    (_FL, _WB): TransitionType.DEFORESTATION,

    # ---- FROM: CROPLAND ----
    (_CL, _FL): TransitionType.REFORESTATION,
    (_CL, _CL): TransitionType.STABLE,
    (_CL, _GL): TransitionType.AGRICULTURAL_ABANDONMENT,
    (_CL, _WL): TransitionType.WETLAND_RESTORATION,
    (_CL, _SL): TransitionType.URBANIZATION,
    (_CL, _OL): TransitionType.AGRICULTURAL_ABANDONMENT,
    (_CL, _OP): TransitionType.COMMODITY_CONVERSION,
    (_CL, _RP): TransitionType.COMMODITY_CONVERSION,
    (_CL, _PF): TransitionType.AFFORESTATION,
    (_CL, _WB): TransitionType.INUNDATION,

    # ---- FROM: GRASSLAND ----
    (_GL, _FL): TransitionType.REFORESTATION,
    (_GL, _CL): TransitionType.AGRICULTURAL_EXPANSION,
    (_GL, _GL): TransitionType.STABLE,
    (_GL, _WL): TransitionType.WETLAND_RESTORATION,
    (_GL, _SL): TransitionType.URBANIZATION,
    (_GL, _OL): TransitionType.NATURAL_SUCCESSION,
    (_GL, _OP): TransitionType.AGRICULTURAL_EXPANSION,
    (_GL, _RP): TransitionType.AGRICULTURAL_EXPANSION,
    (_GL, _PF): TransitionType.AFFORESTATION,
    (_GL, _WB): TransitionType.INUNDATION,

    # ---- FROM: WETLAND ----
    (_WL, _FL): TransitionType.REFORESTATION,
    (_WL, _CL): TransitionType.WETLAND_CONVERSION,
    (_WL, _GL): TransitionType.WETLAND_CONVERSION,
    (_WL, _WL): TransitionType.STABLE,
    (_WL, _SL): TransitionType.URBANIZATION,
    (_WL, _OL): TransitionType.WETLAND_CONVERSION,
    (_WL, _OP): TransitionType.WETLAND_CONVERSION,
    (_WL, _RP): TransitionType.WETLAND_CONVERSION,
    (_WL, _PF): TransitionType.AFFORESTATION,
    (_WL, _WB): TransitionType.INUNDATION,

    # ---- FROM: SETTLEMENT ----
    (_SL, _FL): TransitionType.REFORESTATION,
    (_SL, _CL): TransitionType.LAND_RECLAMATION,
    (_SL, _GL): TransitionType.LAND_RECLAMATION,
    (_SL, _WL): TransitionType.WETLAND_RESTORATION,
    (_SL, _SL): TransitionType.STABLE,
    (_SL, _OL): TransitionType.LAND_RECLAMATION,
    (_SL, _OP): TransitionType.LAND_RECLAMATION,
    (_SL, _RP): TransitionType.LAND_RECLAMATION,
    (_SL, _PF): TransitionType.AFFORESTATION,
    (_SL, _WB): TransitionType.INUNDATION,

    # ---- FROM: OTHER_LAND ----
    (_OL, _FL): TransitionType.AFFORESTATION,
    (_OL, _CL): TransitionType.AGRICULTURAL_EXPANSION,
    (_OL, _GL): TransitionType.NATURAL_SUCCESSION,
    (_OL, _WL): TransitionType.WETLAND_RESTORATION,
    (_OL, _SL): TransitionType.URBANIZATION,
    (_OL, _OL): TransitionType.STABLE,
    (_OL, _OP): TransitionType.AGRICULTURAL_EXPANSION,
    (_OL, _RP): TransitionType.AGRICULTURAL_EXPANSION,
    (_OL, _PF): TransitionType.AFFORESTATION,
    (_OL, _WB): TransitionType.INUNDATION,

    # ---- FROM: OIL_PALM_PLANTATION ----
    (_OP, _FL): TransitionType.REFORESTATION,
    (_OP, _CL): TransitionType.AGRICULTURAL_CONVERSION,
    (_OP, _GL): TransitionType.AGRICULTURAL_ABANDONMENT,
    (_OP, _WL): TransitionType.WETLAND_RESTORATION,
    (_OP, _SL): TransitionType.URBANIZATION,
    (_OP, _OL): TransitionType.AGRICULTURAL_ABANDONMENT,
    (_OP, _OP): TransitionType.STABLE,
    (_OP, _RP): TransitionType.COMMODITY_CONVERSION,
    (_OP, _PF): TransitionType.AFFORESTATION,
    (_OP, _WB): TransitionType.INUNDATION,

    # ---- FROM: RUBBER_PLANTATION ----
    (_RP, _FL): TransitionType.REFORESTATION,
    (_RP, _CL): TransitionType.AGRICULTURAL_CONVERSION,
    (_RP, _GL): TransitionType.AGRICULTURAL_ABANDONMENT,
    (_RP, _WL): TransitionType.WETLAND_RESTORATION,
    (_RP, _SL): TransitionType.URBANIZATION,
    (_RP, _OL): TransitionType.AGRICULTURAL_ABANDONMENT,
    (_RP, _OP): TransitionType.COMMODITY_CONVERSION,
    (_RP, _RP): TransitionType.STABLE,
    (_RP, _PF): TransitionType.AFFORESTATION,
    (_RP, _WB): TransitionType.INUNDATION,

    # ---- FROM: PLANTATION_FOREST ----
    (_PF, _FL): TransitionType.NATURAL_SUCCESSION,
    (_PF, _CL): TransitionType.DEFORESTATION,
    (_PF, _GL): TransitionType.DEFORESTATION,
    (_PF, _WL): TransitionType.DEFORESTATION,
    (_PF, _SL): TransitionType.DEFORESTATION,
    (_PF, _OL): TransitionType.DEFORESTATION,
    (_PF, _OP): TransitionType.DEFORESTATION,
    (_PF, _RP): TransitionType.DEFORESTATION,
    (_PF, _PF): TransitionType.PLANTATION_MANAGEMENT,
    (_PF, _WB): TransitionType.DEFORESTATION,

    # ---- FROM: WATER_BODY ----
    (_WB, _FL): TransitionType.LAND_RECLAMATION,
    (_WB, _CL): TransitionType.LAND_RECLAMATION,
    (_WB, _GL): TransitionType.LAND_RECLAMATION,
    (_WB, _WL): TransitionType.NATURAL_SUCCESSION,
    (_WB, _SL): TransitionType.LAND_RECLAMATION,
    (_WB, _OL): TransitionType.LAND_RECLAMATION,
    (_WB, _OP): TransitionType.LAND_RECLAMATION,
    (_WB, _RP): TransitionType.LAND_RECLAMATION,
    (_WB, _PF): TransitionType.LAND_RECLAMATION,
    (_WB, _WB): TransitionType.STABLE,
}


# ---------------------------------------------------------------------------
# Transition severity classification
# ---------------------------------------------------------------------------
#
# Maps each TransitionType to a severity level and the EUDR article reference.

TRANSITION_SEVERITY: Dict[TransitionType, Dict[str, str]] = {
    TransitionType.DEFORESTATION: {
        "severity": "critical",
        "eudr_article": "Art. 2(1)",
        "description": (
            "Conversion of forest to non-forest use. Directly prohibited "
            "under EUDR Article 3(a) for products placed on or exported "
            "from the Union market after the cutoff date."
        ),
        "action_required": "immediate_block",
    },
    TransitionType.DEGRADATION: {
        "severity": "high",
        "eudr_article": "Art. 2(5)",
        "description": (
            "Structural change to forest land in the form of conversion "
            "of primary or naturally regenerating forest to plantation "
            "forest. Subject to EUDR prohibition."
        ),
        "action_required": "immediate_block",
    },
    TransitionType.REFORESTATION: {
        "severity": "positive",
        "eudr_article": "N/A",
        "description": (
            "Re-establishment of forest on previously forested land. "
            "Positive indicator for compliance but does not reverse "
            "prior deforestation events for EUDR purposes."
        ),
        "action_required": "monitor",
    },
    TransitionType.AFFORESTATION: {
        "severity": "positive",
        "eudr_article": "N/A",
        "description": (
            "Establishment of forest on land not previously forested. "
            "No EUDR compliance concerns."
        ),
        "action_required": "none",
    },
    TransitionType.AGRICULTURAL_EXPANSION: {
        "severity": "medium",
        "eudr_article": "Art. 10(2)",
        "description": (
            "Expansion of agricultural area into non-forest land. "
            "Not deforestation per se, but requires risk assessment "
            "under EUDR Article 10(2) for source area legality."
        ),
        "action_required": "risk_assessment",
    },
    TransitionType.AGRICULTURAL_CONVERSION: {
        "severity": "low",
        "eudr_article": "Art. 10(2)",
        "description": (
            "Conversion between agricultural crop types. Not deforestation "
            "but may require legality verification under EUDR."
        ),
        "action_required": "verify_legality",
    },
    TransitionType.AGRICULTURAL_ABANDONMENT: {
        "severity": "low",
        "eudr_article": "N/A",
        "description": (
            "Abandonment of agricultural land. May indicate natural "
            "regeneration. No EUDR compliance concerns."
        ),
        "action_required": "none",
    },
    TransitionType.URBANIZATION: {
        "severity": "medium",
        "eudr_article": "Art. 2(1)",
        "description": (
            "Conversion to settlement or built-up area. If from forest, "
            "classified as deforestation. Otherwise, requires legality check."
        ),
        "action_required": "verify_legality",
    },
    TransitionType.WETLAND_CONVERSION: {
        "severity": "high",
        "eudr_article": "Art. 10(2)",
        "description": (
            "Drainage or conversion of wetland for agriculture. "
            "Particularly relevant for peatland drainage in palm oil "
            "supply chains. High environmental impact."
        ),
        "action_required": "enhanced_review",
    },
    TransitionType.WETLAND_RESTORATION: {
        "severity": "positive",
        "eudr_article": "N/A",
        "description": (
            "Restoration of wetland ecosystems. Positive environmental "
            "outcome. No EUDR compliance concerns."
        ),
        "action_required": "none",
    },
    TransitionType.COMMODITY_CONVERSION: {
        "severity": "low",
        "eudr_article": "Art. 10(2)",
        "description": (
            "Conversion between commodity crop types on existing "
            "agricultural land. Not deforestation but requires "
            "legality verification."
        ),
        "action_required": "verify_legality",
    },
    TransitionType.PLANTATION_MANAGEMENT: {
        "severity": "excluded",
        "eudr_article": "Art. 2(4)",
        "description": (
            "Conversion between plantation forest types. Explicitly "
            "excluded from deforestation definition per EUDR Art. 2(4)."
        ),
        "action_required": "none",
    },
    TransitionType.STABLE: {
        "severity": "none",
        "eudr_article": "N/A",
        "description": (
            "No land use change detected. Same land use category "
            "at both observation dates."
        ),
        "action_required": "none",
    },
    TransitionType.NATURAL_SUCCESSION: {
        "severity": "low",
        "eudr_article": "N/A",
        "description": (
            "Natural ecological succession between land use types. "
            "No human-driven land use change detected."
        ),
        "action_required": "none",
    },
    TransitionType.INUNDATION: {
        "severity": "low",
        "eudr_article": "N/A",
        "description": (
            "Land covered by water due to flooding, reservoir creation, "
            "or sea level change. Not directly EUDR-relevant."
        ),
        "action_required": "none",
    },
    TransitionType.LAND_RECLAMATION: {
        "severity": "low",
        "eudr_article": "N/A",
        "description": (
            "Reclamation of land from water or degraded state. "
            "No EUDR compliance concerns."
        ),
        "action_required": "none",
    },
    TransitionType.UNCLASSIFIED: {
        "severity": "unknown",
        "eudr_article": "N/A",
        "description": (
            "Transition could not be classified. Requires manual review "
            "and additional data collection."
        ),
        "action_required": "manual_review",
    },
}


# ---------------------------------------------------------------------------
# Verdict determination rules
# ---------------------------------------------------------------------------
#
# Comprehensive rules for CutoffDateVerifier to determine compliance verdict.
# Each rule maps a condition to a ComplianceVerdict with explanation.
#
# Format:
#   (cutoff_class, current_class) -> {
#       "verdict": ComplianceVerdict,
#       "confidence_threshold": float,
#       "explanation": str,
#   }
#
# When confidence is below the threshold, verdict becomes REQUIRES_REVIEW.

VERDICT_DETERMINATION_RULES: Dict[
    Tuple[LandUseCategory, LandUseCategory], Dict[str, Any]
] = {
    # Forest at cutoff, still forest -> COMPLIANT
    (_FL, _FL): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.70,
        "explanation": (
            "Forest land at cutoff date and current observation. "
            "No deforestation detected. EUDR compliant."
        ),
    },
    # Forest at cutoff, now cropland -> NON_COMPLIANT
    (_FL, _CL): {
        "verdict": ComplianceVerdict.NON_COMPLIANT,
        "confidence_threshold": 0.75,
        "explanation": (
            "Forest land at cutoff date converted to cropland. "
            "Post-cutoff deforestation per EUDR Art. 2(1). Non-compliant."
        ),
    },
    # Forest at cutoff, now grassland -> NON_COMPLIANT
    (_FL, _GL): {
        "verdict": ComplianceVerdict.NON_COMPLIANT,
        "confidence_threshold": 0.75,
        "explanation": (
            "Forest land at cutoff date converted to grassland/pasture. "
            "Post-cutoff deforestation per EUDR Art. 2(1). Non-compliant."
        ),
    },
    # Forest at cutoff, now oil palm -> NON_COMPLIANT
    (_FL, _OP): {
        "verdict": ComplianceVerdict.NON_COMPLIANT,
        "confidence_threshold": 0.80,
        "explanation": (
            "Forest land at cutoff date converted to oil palm plantation. "
            "Post-cutoff deforestation per EUDR Art. 2(1). Non-compliant."
        ),
    },
    # Forest at cutoff, now rubber -> NON_COMPLIANT
    (_FL, _RP): {
        "verdict": ComplianceVerdict.NON_COMPLIANT,
        "confidence_threshold": 0.80,
        "explanation": (
            "Forest land at cutoff date converted to rubber plantation. "
            "Post-cutoff deforestation per EUDR Art. 2(1). Non-compliant."
        ),
    },
    # Forest at cutoff, now plantation forest -> NON_COMPLIANT (degradation)
    (_FL, _PF): {
        "verdict": ComplianceVerdict.NON_COMPLIANT,
        "confidence_threshold": 0.80,
        "explanation": (
            "Natural forest at cutoff date converted to plantation forest. "
            "Post-cutoff forest degradation per EUDR Art. 2(5). Non-compliant."
        ),
    },
    # Forest at cutoff, now settlement -> NON_COMPLIANT
    (_FL, _SL): {
        "verdict": ComplianceVerdict.NON_COMPLIANT,
        "confidence_threshold": 0.75,
        "explanation": (
            "Forest land at cutoff date converted to settlement. "
            "Post-cutoff deforestation per EUDR Art. 2(1). Non-compliant."
        ),
    },
    # Forest at cutoff, now other land -> NON_COMPLIANT
    (_FL, _OL): {
        "verdict": ComplianceVerdict.NON_COMPLIANT,
        "confidence_threshold": 0.75,
        "explanation": (
            "Forest land at cutoff date converted to barren/other land. "
            "Post-cutoff deforestation per EUDR Art. 2(1). Non-compliant."
        ),
    },
    # Forest at cutoff, now wetland -> NON_COMPLIANT
    (_FL, _WL): {
        "verdict": ComplianceVerdict.NON_COMPLIANT,
        "confidence_threshold": 0.75,
        "explanation": (
            "Forest land at cutoff date converted to wetland. "
            "Post-cutoff deforestation per EUDR Art. 2(1). Non-compliant."
        ),
    },
    # Forest at cutoff, now water body -> NON_COMPLIANT
    (_FL, _WB): {
        "verdict": ComplianceVerdict.NON_COMPLIANT,
        "confidence_threshold": 0.75,
        "explanation": (
            "Forest land at cutoff date now submerged. "
            "Post-cutoff deforestation per EUDR Art. 2(1). Non-compliant."
        ),
    },

    # Cropland at cutoff, still cropland -> COMPLIANT
    (_CL, _CL): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.70,
        "explanation": (
            "Cropland at cutoff date and current observation. "
            "No deforestation. EUDR compliant."
        ),
    },
    # Cropland at cutoff, now forest -> COMPLIANT
    (_CL, _FL): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.70,
        "explanation": (
            "Cropland at cutoff date has reforested. "
            "Positive land use change. EUDR compliant."
        ),
    },
    # Cropland at cutoff, any non-forest -> COMPLIANT (no deforestation)
    (_CL, _GL): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.65,
        "explanation": (
            "Cropland at cutoff date converted to grassland. "
            "No deforestation involved. EUDR compliant."
        ),
    },
    (_CL, _WL): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.65,
        "explanation": (
            "Cropland at cutoff date converted to wetland. "
            "No deforestation. EUDR compliant."
        ),
    },
    (_CL, _SL): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.65,
        "explanation": (
            "Cropland at cutoff date urbanized. "
            "No deforestation. EUDR compliant."
        ),
    },
    (_CL, _OL): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.65,
        "explanation": (
            "Cropland at cutoff date abandoned. "
            "No deforestation. EUDR compliant."
        ),
    },
    (_CL, _OP): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.70,
        "explanation": (
            "Cropland at cutoff date converted to oil palm. "
            "No deforestation (was already non-forest). EUDR compliant."
        ),
    },
    (_CL, _RP): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.70,
        "explanation": (
            "Cropland at cutoff date converted to rubber. "
            "No deforestation (was already non-forest). EUDR compliant."
        ),
    },
    (_CL, _PF): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.70,
        "explanation": (
            "Cropland at cutoff date afforested with plantation. "
            "Positive land use change. EUDR compliant."
        ),
    },
    (_CL, _WB): {
        "verdict": ComplianceVerdict.COMPLIANT,
        "confidence_threshold": 0.65,
        "explanation": (
            "Cropland at cutoff date now inundated. "
            "No deforestation. EUDR compliant."
        ),
    },

    # Grassland at cutoff -> any outcome: COMPLIANT (was not forest)
    (_GL, _FL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                 "explanation": "Grassland reforested. EUDR compliant."},
    (_GL, _CL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Grassland converted to cropland. No deforestation. EUDR compliant."},
    (_GL, _GL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Grassland stable. EUDR compliant."},
    (_GL, _WL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Grassland to wetland. EUDR compliant."},
    (_GL, _SL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Grassland urbanized. EUDR compliant."},
    (_GL, _OL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Grassland degraded to other land. EUDR compliant."},
    (_GL, _OP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Grassland to oil palm. No deforestation. EUDR compliant."},
    (_GL, _RP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Grassland to rubber. No deforestation. EUDR compliant."},
    (_GL, _PF): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Grassland afforested. EUDR compliant."},
    (_GL, _WB): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Grassland inundated. EUDR compliant."},

    # Wetland at cutoff -> any outcome: COMPLIANT (was not forest)
    (_WL, _FL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Wetland reforested. EUDR compliant."},
    (_WL, _CL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Wetland converted to cropland. No deforestation. EUDR compliant."},
    (_WL, _GL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Wetland drained to grassland. EUDR compliant."},
    (_WL, _WL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Wetland stable. EUDR compliant."},
    (_WL, _SL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Wetland urbanized. EUDR compliant."},
    (_WL, _OL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Wetland drained to other land. EUDR compliant."},
    (_WL, _OP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Wetland to oil palm. No deforestation. EUDR compliant."},
    (_WL, _RP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Wetland to rubber. No deforestation. EUDR compliant."},
    (_WL, _PF): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Wetland afforested. EUDR compliant."},
    (_WL, _WB): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Wetland inundated. EUDR compliant."},

    # Settlement at cutoff -> any outcome: COMPLIANT
    (_SL, _FL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Settlement reforested. EUDR compliant."},
    (_SL, _CL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Settlement to cropland. EUDR compliant."},
    (_SL, _GL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Settlement to grassland. EUDR compliant."},
    (_SL, _WL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Settlement to wetland. EUDR compliant."},
    (_SL, _SL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Settlement stable. EUDR compliant."},
    (_SL, _OL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Settlement to other land. EUDR compliant."},
    (_SL, _OP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Settlement to oil palm. EUDR compliant."},
    (_SL, _RP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Settlement to rubber. EUDR compliant."},
    (_SL, _PF): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Settlement afforested. EUDR compliant."},
    (_SL, _WB): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Settlement inundated. EUDR compliant."},

    # Other land at cutoff -> any outcome: COMPLIANT
    (_OL, _FL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Other land afforested. EUDR compliant."},
    (_OL, _CL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Other land to cropland. EUDR compliant."},
    (_OL, _GL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Other land to grassland. EUDR compliant."},
    (_OL, _WL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Other land to wetland. EUDR compliant."},
    (_OL, _SL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Other land urbanized. EUDR compliant."},
    (_OL, _OL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Other land stable. EUDR compliant."},
    (_OL, _OP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Other land to oil palm. EUDR compliant."},
    (_OL, _RP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Other land to rubber. EUDR compliant."},
    (_OL, _PF): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Other land afforested. EUDR compliant."},
    (_OL, _WB): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Other land inundated. EUDR compliant."},

    # Oil palm at cutoff -> any outcome: COMPLIANT (was already non-forest)
    (_OP, _FL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Oil palm reforested. EUDR compliant."},
    (_OP, _CL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Oil palm to cropland. EUDR compliant."},
    (_OP, _GL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Oil palm abandoned. EUDR compliant."},
    (_OP, _WL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Oil palm to wetland. EUDR compliant."},
    (_OP, _SL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Oil palm urbanized. EUDR compliant."},
    (_OP, _OL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Oil palm abandoned. EUDR compliant."},
    (_OP, _OP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Oil palm stable. EUDR compliant."},
    (_OP, _RP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Oil palm to rubber. EUDR compliant."},
    (_OP, _PF): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Oil palm to plantation forest. EUDR compliant."},
    (_OP, _WB): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Oil palm inundated. EUDR compliant."},

    # Rubber at cutoff -> any outcome: COMPLIANT (was already non-forest)
    (_RP, _FL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Rubber reforested. EUDR compliant."},
    (_RP, _CL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Rubber to cropland. EUDR compliant."},
    (_RP, _GL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Rubber abandoned. EUDR compliant."},
    (_RP, _WL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Rubber to wetland. EUDR compliant."},
    (_RP, _SL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Rubber urbanized. EUDR compliant."},
    (_RP, _OL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Rubber abandoned. EUDR compliant."},
    (_RP, _OP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Rubber to oil palm. EUDR compliant."},
    (_RP, _RP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Rubber stable. EUDR compliant."},
    (_RP, _PF): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Rubber to plantation forest. EUDR compliant."},
    (_RP, _WB): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Rubber inundated. EUDR compliant."},

    # Plantation forest at cutoff -> non-forest: NON_COMPLIANT (still forest)
    (_PF, _FL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.70,
                  "explanation": "Plantation forest to natural forest succession. EUDR compliant."},
    (_PF, _CL): {"verdict": ComplianceVerdict.NON_COMPLIANT, "confidence_threshold": 0.75,
                  "explanation": "Plantation forest to cropland. Post-cutoff deforestation. Non-compliant."},
    (_PF, _GL): {"verdict": ComplianceVerdict.NON_COMPLIANT, "confidence_threshold": 0.75,
                  "explanation": "Plantation forest to grassland. Post-cutoff deforestation. Non-compliant."},
    (_PF, _WL): {"verdict": ComplianceVerdict.NON_COMPLIANT, "confidence_threshold": 0.75,
                  "explanation": "Plantation forest to wetland. Post-cutoff deforestation. Non-compliant."},
    (_PF, _SL): {"verdict": ComplianceVerdict.NON_COMPLIANT, "confidence_threshold": 0.75,
                  "explanation": "Plantation forest urbanized. Post-cutoff deforestation. Non-compliant."},
    (_PF, _OL): {"verdict": ComplianceVerdict.NON_COMPLIANT, "confidence_threshold": 0.75,
                  "explanation": "Plantation forest to barren land. Post-cutoff deforestation. Non-compliant."},
    (_PF, _OP): {"verdict": ComplianceVerdict.NON_COMPLIANT, "confidence_threshold": 0.80,
                  "explanation": "Plantation forest to oil palm. Post-cutoff deforestation. Non-compliant."},
    (_PF, _RP): {"verdict": ComplianceVerdict.NON_COMPLIANT, "confidence_threshold": 0.80,
                  "explanation": "Plantation forest to rubber. Post-cutoff deforestation. Non-compliant."},
    (_PF, _PF): {"verdict": ComplianceVerdict.EXCLUDED, "confidence_threshold": 0.70,
                  "explanation": "Plantation forest to plantation forest. Excluded per Art. 2(4). Compliant."},
    (_PF, _WB): {"verdict": ComplianceVerdict.NON_COMPLIANT, "confidence_threshold": 0.75,
                  "explanation": "Plantation forest inundated. Post-cutoff deforestation. Non-compliant."},

    # Water body at cutoff -> any outcome: COMPLIANT
    (_WB, _FL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Water body to forest. Land reclamation. EUDR compliant."},
    (_WB, _CL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Water body to cropland. Land reclamation. EUDR compliant."},
    (_WB, _GL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Water body to grassland. Land reclamation. EUDR compliant."},
    (_WB, _WL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Water body to wetland. Natural succession. EUDR compliant."},
    (_WB, _SL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Water body to settlement. Land reclamation. EUDR compliant."},
    (_WB, _OL): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Water body to other land. Land reclamation. EUDR compliant."},
    (_WB, _OP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Water body to oil palm. Land reclamation. EUDR compliant."},
    (_WB, _RP): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Water body to rubber. Land reclamation. EUDR compliant."},
    (_WB, _PF): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Water body to plantation forest. Land reclamation. EUDR compliant."},
    (_WB, _WB): {"verdict": ComplianceVerdict.COMPLIANT, "confidence_threshold": 0.65,
                  "explanation": "Water body stable. EUDR compliant."},
}


# ---------------------------------------------------------------------------
# Commodity conversion rules
# ---------------------------------------------------------------------------
#
# Commodity-specific rules that define which transitions constitute
# EUDR-relevant commodity deforestation. Keys are commodity names, values
# are sets of (from_class, to_class) tuples that trigger non-compliance.

COMMODITY_CONVERSION_RULES: Dict[str, Set[Tuple[LandUseCategory, LandUseCategory]]] = {
    "palm_oil": {
        (_FL, _OP), (_FL, _CL), (_PF, _OP), (_PF, _CL),
        (_WL, _OP),  # Peatland drainage for palm oil
    },
    "rubber": {
        (_FL, _RP), (_FL, _CL), (_PF, _RP), (_PF, _CL),
    },
    "soya": {
        (_FL, _CL), (_PF, _CL), (_GL, _CL),
    },
    "cattle": {
        (_FL, _GL), (_PF, _GL), (_FL, _CL),
    },
    "cocoa": {
        (_FL, _CL), (_PF, _CL),
    },
    "coffee": {
        (_FL, _CL), (_PF, _CL),
    },
    "wood": {
        (_FL, _OL), (_FL, _CL), (_FL, _GL),
        (_PF, _OL), (_PF, _CL), (_PF, _GL),
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def classify_transition(
    from_class: LandUseCategory,
    to_class: LandUseCategory,
) -> TransitionType:
    """Classify a land use transition using the comprehensive rules matrix.

    Looks up the (from_class, to_class) pair in the
    TRANSITION_CLASSIFICATION_RULES dictionary. Returns UNCLASSIFIED
    if the combination is not explicitly defined.

    Args:
        from_class: Source land use category.
        to_class: Destination land use category.

    Returns:
        TransitionType classification for the transition.
    """
    return TRANSITION_CLASSIFICATION_RULES.get(
        (from_class, to_class),
        TransitionType.UNCLASSIFIED,
    )


def is_deforestation(
    from_class: LandUseCategory,
    to_class: LandUseCategory,
) -> bool:
    """Determine whether a transition constitutes deforestation.

    A transition is deforestation if its classification type is
    DEFORESTATION per the rules matrix.

    Args:
        from_class: Source land use category.
        to_class: Destination land use category.

    Returns:
        True if the transition is classified as deforestation.
    """
    return classify_transition(from_class, to_class) == TransitionType.DEFORESTATION


def is_degradation(
    from_class: LandUseCategory,
    to_class: LandUseCategory,
) -> bool:
    """Determine whether a transition constitutes forest degradation.

    A transition is degradation if its classification type is
    DEGRADATION per the rules matrix.

    Args:
        from_class: Source land use category.
        to_class: Destination land use category.

    Returns:
        True if the transition is classified as degradation.
    """
    return classify_transition(from_class, to_class) == TransitionType.DEGRADATION


def determine_verdict(
    cutoff_class: LandUseCategory,
    current_class: LandUseCategory,
    confidence: float = 1.0,
) -> Tuple[ComplianceVerdict, str]:
    """Determine the EUDR compliance verdict for a land use transition.

    Uses the VERDICT_DETERMINATION_RULES to assign a compliance verdict.
    If the classification confidence is below the rule's threshold,
    the verdict is overridden to REQUIRES_REVIEW.

    Args:
        cutoff_class: Land use category at the EUDR cutoff date (2020-12-31).
        current_class: Current land use category at the observation date.
        confidence: Classification confidence score in [0.0, 1.0].
            Defaults to 1.0 (full confidence).

    Returns:
        Tuple of (ComplianceVerdict, explanation_string).
    """
    rule = VERDICT_DETERMINATION_RULES.get((cutoff_class, current_class))

    if rule is None:
        return (
            ComplianceVerdict.INSUFFICIENT_DATA,
            f"No verdict rule for transition ({cutoff_class.value} -> "
            f"{current_class.value}). Manual review required.",
        )

    verdict = rule["verdict"]
    threshold = rule.get("confidence_threshold", 0.70)
    explanation = rule.get("explanation", "")

    if confidence < threshold:
        return (
            ComplianceVerdict.REQUIRES_REVIEW,
            f"Confidence ({confidence:.2f}) below threshold "
            f"({threshold:.2f}). {explanation} Review required.",
        )

    return (verdict, explanation)


def get_eudr_article(transition_type: TransitionType) -> str:
    """Retrieve the EUDR article reference for a transition type.

    Args:
        transition_type: The classified transition type.

    Returns:
        EUDR article string (e.g., 'Art. 2(1)') or 'N/A' if not
        directly referenced in EUDR.
    """
    severity_info = TRANSITION_SEVERITY.get(transition_type)
    if severity_info is None:
        return "N/A"
    return severity_info.get("eudr_article", "N/A")


def get_transition_severity(transition_type: TransitionType) -> str:
    """Retrieve the severity level for a transition type.

    Args:
        transition_type: The classified transition type.

    Returns:
        Severity string: 'critical', 'high', 'medium', 'low',
        'positive', 'excluded', 'none', or 'unknown'.
    """
    severity_info = TRANSITION_SEVERITY.get(transition_type)
    if severity_info is None:
        return "unknown"
    return severity_info.get("severity", "unknown")


def get_commodity_deforestation_transitions(commodity: str) -> Set[Tuple[LandUseCategory, LandUseCategory]]:
    """Retrieve the set of deforestation transitions for a specific commodity.

    Args:
        commodity: EUDR commodity name (lowercase).

    Returns:
        Set of (from_class, to_class) tuples that constitute
        EUDR-relevant deforestation for the commodity. Empty set if
        commodity is not recognized.
    """
    return COMMODITY_CONVERSION_RULES.get(commodity.lower(), set())


def is_commodity_deforestation(
    commodity: str,
    from_class: LandUseCategory,
    to_class: LandUseCategory,
) -> bool:
    """Determine if a transition is commodity-specific deforestation.

    Args:
        commodity: EUDR commodity name (lowercase).
        from_class: Source land use category.
        to_class: Destination land use category.

    Returns:
        True if the transition constitutes EUDR-relevant deforestation
        for the specified commodity.
    """
    rules = COMMODITY_CONVERSION_RULES.get(commodity.lower(), set())
    return (from_class, to_class) in rules


def get_all_deforestation_transitions() -> List[Tuple[LandUseCategory, LandUseCategory]]:
    """Return all transitions classified as deforestation.

    Returns:
        List of (from_class, to_class) tuples classified as DEFORESTATION.
    """
    return [
        pair for pair, ttype in TRANSITION_CLASSIFICATION_RULES.items()
        if ttype == TransitionType.DEFORESTATION
    ]


def get_all_non_compliant_verdicts() -> List[Tuple[LandUseCategory, LandUseCategory]]:
    """Return all cutoff->current transitions that yield NON_COMPLIANT verdict.

    Returns:
        List of (cutoff_class, current_class) tuples that result in
        NON_COMPLIANT verdict.
    """
    return [
        pair for pair, rule in VERDICT_DETERMINATION_RULES.items()
        if rule.get("verdict") == ComplianceVerdict.NON_COMPLIANT
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TransitionType",
    "ComplianceVerdict",
    "TRANSITION_CLASSIFICATION_RULES",
    "TRANSITION_SEVERITY",
    "VERDICT_DETERMINATION_RULES",
    "COMMODITY_CONVERSION_RULES",
    "classify_transition",
    "is_deforestation",
    "is_degradation",
    "determine_verdict",
    "get_eudr_article",
    "get_transition_severity",
    "get_commodity_deforestation_transitions",
    "is_commodity_deforestation",
    "get_all_deforestation_transitions",
    "get_all_non_compliant_verdicts",
]
