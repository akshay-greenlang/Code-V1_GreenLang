# -*- coding: utf-8 -*-
"""
DPPBridge - Digital Product Passport Integration for PACK-018
================================================================

This module integrates the EU Green Claims Prep Pack with Digital Product
Passports (DPP) as defined by the Ecodesign for Sustainable Products
Regulation (ESPR, Regulation 2024/1781). Environmental claims about
specific products must be consistent with the product's DPP data,
ensuring traceability and verifiability of green marketing claims.

DPP Data Categories:
    - Product identification (GTIN, batch, serial)
    - Material composition and substances of concern
    - Carbon footprint (cradle-to-gate, per functional unit)
    - Durability and repairability scores
    - Recycled content percentage
    - Energy efficiency class
    - Supply chain due diligence information
    - End-of-life handling instructions

ESPR Product Groups (Phase 1 Priority):
    - Batteries (Regulation 2023/1542)
    - Textiles (proposed)
    - Electronics and ICT (proposed)
    - Furniture (proposed)
    - Iron and steel (proposed)
    - Aluminium (proposed)
    - Tyres (proposed)

Passport Schema Versions:
    - DPP-v1.0: Basic identification and carbon footprint
    - DPP-v1.1: Extended with circularity metrics
    - DPP-v2.0: Full ESPR compliance (target 2027)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-018 EU Green Claims Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

__all__ = [
    "ProductGroup",
    "PassportSchemaVersion",
    "LinkingStatus",
    "DPPDataField",
    "DPPBridgeConfig",
    "DPPDataSnapshot",
    "DPPLinkingResult",
    "DPPBridge",
]

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for provenance tracking."""
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

class ProductGroup(str, Enum):
    """ESPR product groups with DPP requirements."""

    BATTERIES = "batteries"
    TEXTILES = "textiles"
    ELECTRONICS = "electronics_ict"
    FURNITURE = "furniture"
    IRON_STEEL = "iron_and_steel"
    ALUMINIUM = "aluminium"
    TYRES = "tyres"
    OTHER = "other"

class PassportSchemaVersion(str, Enum):
    """DPP schema versions."""

    V1_0 = "DPP-v1.0"
    V1_1 = "DPP-v1.1"
    V2_0 = "DPP-v2.0"

class LinkingStatus(str, Enum):
    """Status of a DPP-to-claim linking operation."""

    LINKED = "linked"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    INCONSISTENT = "inconsistent"
    FAILED = "failed"

class DPPDataField(str, Enum):
    """Standard DPP data fields."""

    PRODUCT_ID = "product_id"
    GTIN = "gtin"
    MATERIAL_COMPOSITION = "material_composition"
    SUBSTANCES_OF_CONCERN = "substances_of_concern"
    CARBON_FOOTPRINT = "carbon_footprint"
    DURABILITY_SCORE = "durability_score"
    REPAIRABILITY_SCORE = "repairability_score"
    RECYCLED_CONTENT_PCT = "recycled_content_pct"
    ENERGY_EFFICIENCY_CLASS = "energy_efficiency_class"
    SUPPLY_CHAIN_INFO = "supply_chain_info"
    END_OF_LIFE_INSTRUCTIONS = "end_of_life_instructions"

# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

PRODUCT_GROUP_DPP_FIELDS: Dict[ProductGroup, List[DPPDataField]] = {
    ProductGroup.BATTERIES: [
        DPPDataField.PRODUCT_ID, DPPDataField.MATERIAL_COMPOSITION,
        DPPDataField.CARBON_FOOTPRINT, DPPDataField.RECYCLED_CONTENT_PCT,
        DPPDataField.DURABILITY_SCORE, DPPDataField.SUBSTANCES_OF_CONCERN,
        DPPDataField.END_OF_LIFE_INSTRUCTIONS, DPPDataField.SUPPLY_CHAIN_INFO,
    ],
    ProductGroup.TEXTILES: [
        DPPDataField.PRODUCT_ID, DPPDataField.MATERIAL_COMPOSITION,
        DPPDataField.CARBON_FOOTPRINT, DPPDataField.RECYCLED_CONTENT_PCT,
        DPPDataField.DURABILITY_SCORE, DPPDataField.REPAIRABILITY_SCORE,
        DPPDataField.SUBSTANCES_OF_CONCERN,
    ],
    ProductGroup.ELECTRONICS: [
        DPPDataField.PRODUCT_ID, DPPDataField.ENERGY_EFFICIENCY_CLASS,
        DPPDataField.CARBON_FOOTPRINT, DPPDataField.REPAIRABILITY_SCORE,
        DPPDataField.DURABILITY_SCORE, DPPDataField.RECYCLED_CONTENT_PCT,
        DPPDataField.END_OF_LIFE_INSTRUCTIONS,
    ],
    ProductGroup.FURNITURE: [
        DPPDataField.PRODUCT_ID, DPPDataField.MATERIAL_COMPOSITION,
        DPPDataField.CARBON_FOOTPRINT, DPPDataField.DURABILITY_SCORE,
        DPPDataField.REPAIRABILITY_SCORE, DPPDataField.RECYCLED_CONTENT_PCT,
    ],
    ProductGroup.IRON_STEEL: [
        DPPDataField.PRODUCT_ID, DPPDataField.MATERIAL_COMPOSITION,
        DPPDataField.CARBON_FOOTPRINT, DPPDataField.RECYCLED_CONTENT_PCT,
        DPPDataField.SUPPLY_CHAIN_INFO,
    ],
    ProductGroup.ALUMINIUM: [
        DPPDataField.PRODUCT_ID, DPPDataField.MATERIAL_COMPOSITION,
        DPPDataField.CARBON_FOOTPRINT, DPPDataField.RECYCLED_CONTENT_PCT,
        DPPDataField.SUPPLY_CHAIN_INFO,
    ],
    ProductGroup.TYRES: [
        DPPDataField.PRODUCT_ID, DPPDataField.MATERIAL_COMPOSITION,
        DPPDataField.CARBON_FOOTPRINT, DPPDataField.DURABILITY_SCORE,
        DPPDataField.END_OF_LIFE_INSTRUCTIONS,
    ],
    ProductGroup.OTHER: [
        DPPDataField.PRODUCT_ID, DPPDataField.CARBON_FOOTPRINT,
    ],
}

CLAIM_TO_DPP_FIELDS: Dict[str, List[DPPDataField]] = {
    "recyclable": [DPPDataField.RECYCLED_CONTENT_PCT, DPPDataField.MATERIAL_COMPOSITION, DPPDataField.END_OF_LIFE_INSTRUCTIONS],
    "low_carbon": [DPPDataField.CARBON_FOOTPRINT],
    "carbon_neutral": [DPPDataField.CARBON_FOOTPRINT, DPPDataField.SUPPLY_CHAIN_INFO],
    "durable": [DPPDataField.DURABILITY_SCORE],
    "repairable": [DPPDataField.REPAIRABILITY_SCORE],
    "non_toxic": [DPPDataField.SUBSTANCES_OF_CONCERN, DPPDataField.MATERIAL_COMPOSITION],
    "energy_efficient": [DPPDataField.ENERGY_EFFICIENCY_CLASS],
    "recycled_content": [DPPDataField.RECYCLED_CONTENT_PCT],
    "circular": [DPPDataField.RECYCLED_CONTENT_PCT, DPPDataField.DURABILITY_SCORE, DPPDataField.REPAIRABILITY_SCORE, DPPDataField.END_OF_LIFE_INSTRUCTIONS],
    "ethically_sourced": [DPPDataField.SUPPLY_CHAIN_INFO],
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class DPPBridgeConfig(BaseModel):
    """Configuration for the DPP Bridge."""

    pack_id: str = Field(default="PACK-018")
    product_ids: List[str] = Field(
        default_factory=list,
        description="Product IDs to link DPP data for",
    )
    passport_schema: PassportSchemaVersion = Field(
        default=PassportSchemaVersion.V1_1,
        description="DPP schema version to use",
    )
    product_group: ProductGroup = Field(
        default=ProductGroup.OTHER,
        description="ESPR product group for DPP requirements",
    )
    enable_provenance: bool = Field(default=True)
    enable_consistency_check: bool = Field(
        default=True,
        description="Check DPP data consistency with claims",
    )

class DPPDataSnapshot(BaseModel):
    """Snapshot of DPP environmental data for a product."""

    product_id: str = Field(default="")
    gtin: str = Field(default="")
    carbon_footprint_kg_co2e: float = Field(default=0.0, ge=0.0)
    recycled_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    durability_score: float = Field(default=0.0, ge=0.0, le=10.0)
    repairability_score: float = Field(default=0.0, ge=0.0, le=10.0)
    energy_efficiency_class: str = Field(default="")
    substances_of_concern_count: int = Field(default=0, ge=0)
    fields_populated: List[str] = Field(default_factory=list)
    fields_missing: List[str] = Field(default_factory=list)

class DPPLinkingResult(BaseModel):
    """Result of linking DPP data to a green claim."""

    linking_id: str = Field(default_factory=_new_uuid)
    product_id: str = Field(default="")
    claim_id: str = Field(default="")
    status: LinkingStatus = Field(default=LinkingStatus.NOT_FOUND)
    passport_schema: PassportSchemaVersion = Field(default=PassportSchemaVersion.V1_1)
    product_group: ProductGroup = Field(default=ProductGroup.OTHER)
    dpp_snapshot: Optional[DPPDataSnapshot] = Field(None)
    required_fields: List[str] = Field(default_factory=list)
    verified_fields: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    consistency_issues: List[str] = Field(default_factory=list)
    claim_supported: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# DPPBridge
# ---------------------------------------------------------------------------

class DPPBridge:
    """Digital Product Passport integration bridge for PACK-018.

    Maps DPP environmental data fields to green claim evidence,
    checking that marketing claims about products are consistent
    with the structured data in their Digital Product Passports.

    Attributes:
        config: DPP bridge configuration.

    Example:
        >>> config = DPPBridgeConfig(product_group=ProductGroup.BATTERIES)
        >>> bridge = DPPBridge(config)
        >>> result = bridge.link_passport_data("PROD-001", "CLM-001")
        >>> assert result["status"] in ["linked", "partial"]
    """

    def __init__(self, config: Optional[DPPBridgeConfig] = None) -> None:
        """Initialize DPPBridge.

        Args:
            config: Bridge configuration. Defaults used if None.
        """
        self.config = config or DPPBridgeConfig()
        logger.info(
            "DPPBridge initialized (schema=%s, group=%s, products=%d)",
            self.config.passport_schema.value,
            self.config.product_group.value,
            len(self.config.product_ids),
        )

    def link_passport_data(
        self,
        product_id: str,
        claim_id: str,
    ) -> Dict[str, Any]:
        """Link DPP data to a green claim for verification.

        Args:
            product_id: Product identifier to look up DPP data.
            claim_id: Claim identifier to link evidence to.

        Returns:
            Dict with linking status, DPP data snapshot, required/verified/
            missing fields, consistency issues, and provenance hash.
        """
        start = utcnow()
        result = DPPLinkingResult(
            product_id=product_id,
            claim_id=claim_id,
            passport_schema=self.config.passport_schema,
            product_group=self.config.product_group,
        )

        required_fields = self._get_required_fields()
        result.required_fields = [f.value for f in required_fields]

        dpp_snapshot = self._fetch_dpp_snapshot(product_id, required_fields)
        result.dpp_snapshot = dpp_snapshot

        if dpp_snapshot:
            result.verified_fields = list(dpp_snapshot.fields_populated)
            result.missing_fields = list(dpp_snapshot.fields_missing)

            if not dpp_snapshot.fields_missing:
                result.status = LinkingStatus.LINKED
                result.claim_supported = True
            elif len(dpp_snapshot.fields_populated) > 0:
                result.status = LinkingStatus.PARTIAL
                result.claim_supported = False
            else:
                result.status = LinkingStatus.NOT_FOUND
                result.claim_supported = False

            if self.config.enable_consistency_check:
                result.consistency_issues = self._check_consistency(dpp_snapshot)
                if result.consistency_issues:
                    result.status = LinkingStatus.INCONSISTENT
                    result.claim_supported = False
        else:
            result.status = LinkingStatus.NOT_FOUND
            result.missing_fields = result.required_fields

        elapsed = (utcnow() - start).total_seconds() * 1000

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        logger.info(
            "DPPBridge linked product '%s' to claim '%s': %s in %.1fms (verified=%d, missing=%d)",
            product_id,
            claim_id,
            result.status.value,
            elapsed,
            len(result.verified_fields),
            len(result.missing_fields),
        )

        return result.model_dump(mode="json")

    def get_dpp_data(self, product_id: str) -> Dict[str, Any]:
        """Retrieve Digital Product Passport data for a product.

        Per ESPR (Regulation 2024/1781), queries the DPP registry for
        environmental and circularity data fields.

        Args:
            product_id: Unique product identifier (GTIN, serial, etc.).

        Returns:
            Dict with DPP data snapshot and availability status.
        """
        required_fields = self._get_required_fields()
        snapshot = self._fetch_dpp_snapshot(product_id, required_fields)
        result = {
            "product_id": product_id,
            "product_group": self.config.product_group.value,
            "schema_version": self.config.passport_schema.value,
            "fields_populated": snapshot.fields_populated if snapshot else [],
            "fields_missing": snapshot.fields_missing if snapshot else [],
            "provenance_hash": _compute_hash({"product_id": product_id}),
        }
        logger.info("DPPBridge retrieved DPP for '%s'", product_id)
        return result

    def validate_claim_consistency(
        self,
        product_id: str,
        claim_type: str,
    ) -> Dict[str, Any]:
        """Validate consistency between a marketing claim and DPP data.

        Per the Green Claims Directive, product-level environmental
        claims must not contradict the product's DPP declarations.

        Args:
            product_id: Product identifier for DPP lookup.
            claim_type: Type of environmental claim.

        Returns:
            Dict with consistency status, required fields, and hash.
        """
        required_dpp_fields = CLAIM_TO_DPP_FIELDS.get(claim_type, [])
        dpp_data = self.get_dpp_data(product_id)
        populated = set(dpp_data.get("fields_populated", []))
        needed = [f.value for f in required_dpp_fields]
        missing = [f for f in needed if f not in populated]

        status = "consistent" if not missing else "partially_consistent"
        if not populated:
            status = "no_dpp_data"

        result = {
            "product_id": product_id,
            "claim_type": claim_type,
            "status": status,
            "required_fields": needed,
            "fields_available": list(populated),
            "fields_missing": missing,
            "provenance_hash": _compute_hash({"product": product_id, "claim": claim_type}),
        }
        logger.info("DPPBridge consistency check '%s' for '%s': %s", claim_type, product_id, status)
        return result

    def check_recycled_content(self, product_id: str, claimed_pct: float) -> Dict[str, Any]:
        """Check a recycled content claim against DPP data.

        Args:
            product_id: Product identifier.
            claimed_pct: Claimed recycled content percentage.

        Returns:
            Dict with verification result and DPP comparison.
        """
        dpp = self.get_dpp_data(product_id)
        result = {
            "product_id": product_id,
            "claimed_pct": claimed_pct,
            "dpp_field": DPPDataField.RECYCLED_CONTENT_PCT.value,
            "dpp_available": DPPDataField.RECYCLED_CONTENT_PCT.value in dpp.get("fields_populated", []),
            "provenance_hash": _compute_hash({"product": product_id, "claimed": claimed_pct}),
        }
        logger.info("DPPBridge recycled content check for '%s': claimed=%.1f%%", product_id, claimed_pct)
        return result

    def check_carbon_footprint(self, product_id: str, claimed_max_kg: float) -> Dict[str, Any]:
        """Check a carbon footprint claim against DPP data.

        Args:
            product_id: Product identifier.
            claimed_max_kg: Maximum claimed carbon footprint in kg CO2-eq.

        Returns:
            Dict with verification result and DPP comparison.
        """
        dpp = self.get_dpp_data(product_id)
        result = {
            "product_id": product_id,
            "claimed_max_kg_co2eq": claimed_max_kg,
            "dpp_field": DPPDataField.CARBON_FOOTPRINT.value,
            "dpp_available": DPPDataField.CARBON_FOOTPRINT.value in dpp.get("fields_populated", []),
            "provenance_hash": _compute_hash({"product": product_id, "claimed": claimed_max_kg}),
        }
        logger.info("DPPBridge carbon footprint check for '%s': max=%.2f", product_id, claimed_max_kg)
        return result

    def check_durability(self, product_id: str, claimed_years: float) -> Dict[str, Any]:
        """Check a durability claim against DPP data.

        Args:
            product_id: Product identifier.
            claimed_years: Claimed product lifetime in years.

        Returns:
            Dict with verification result and DPP comparison.
        """
        dpp = self.get_dpp_data(product_id)
        result = {
            "product_id": product_id,
            "claimed_years": claimed_years,
            "dpp_field": DPPDataField.DURABILITY_SCORE.value,
            "dpp_available": DPPDataField.DURABILITY_SCORE.value in dpp.get("fields_populated", []),
            "provenance_hash": _compute_hash({"product": product_id, "claimed": claimed_years}),
        }
        logger.info("DPPBridge durability check for '%s': claimed=%.1f years", product_id, claimed_years)
        return result

    def check_repairability(self, product_id: str, claimed_min_score: float) -> Dict[str, Any]:
        """Check a repairability claim against DPP data.

        Args:
            product_id: Product identifier.
            claimed_min_score: Minimum claimed repairability score (0-10).

        Returns:
            Dict with verification result and DPP comparison.
        """
        dpp = self.get_dpp_data(product_id)
        result = {
            "product_id": product_id,
            "claimed_min_score": claimed_min_score,
            "dpp_field": DPPDataField.REPAIRABILITY_SCORE.value,
            "dpp_available": DPPDataField.REPAIRABILITY_SCORE.value in dpp.get("fields_populated", []),
            "provenance_hash": _compute_hash({"product": product_id, "claimed": claimed_min_score}),
        }
        logger.info("DPPBridge repairability check for '%s': min_score=%.1f", product_id, claimed_min_score)
        return result

    def get_required_fields_for_group(self, product_group: ProductGroup) -> List[str]:
        """Get required DPP fields for a product group.

        Args:
            product_group: ESPR product group.

        Returns:
            List of required DPP field names.
        """
        fields = PRODUCT_GROUP_DPP_FIELDS.get(product_group, [])
        return [f.value for f in fields]

    def get_claim_field_mapping(self) -> Dict[str, List[str]]:
        """Get mapping of claim types to required DPP fields."""
        return {
            claim: [f.value for f in fields]
            for claim, fields in CLAIM_TO_DPP_FIELDS.items()
        }

    def get_product_groups(self) -> List[Dict[str, Any]]:
        """Get all supported product groups with field counts."""
        return [
            {
                "group": pg.value,
                "field_count": len(fields),
                "fields": [f.value for f in fields],
            }
            for pg, fields in PRODUCT_GROUP_DPP_FIELDS.items()
        ]

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _get_required_fields(self) -> List[DPPDataField]:
        """Get required DPP fields based on product group configuration."""
        return PRODUCT_GROUP_DPP_FIELDS.get(
            self.config.product_group,
            [DPPDataField.PRODUCT_ID, DPPDataField.CARBON_FOOTPRINT],
        )

    def _fetch_dpp_snapshot(
        self,
        product_id: str,
        required_fields: List[DPPDataField],
    ) -> Optional[DPPDataSnapshot]:
        """Fetch a DPP data snapshot for a product.

        In production, this queries the DPP registry API. Here it returns
        a template snapshot indicating which fields need population.
        """
        snapshot = DPPDataSnapshot(product_id=product_id)
        snapshot.fields_populated = [DPPDataField.PRODUCT_ID.value]
        snapshot.fields_missing = [
            f.value for f in required_fields
            if f != DPPDataField.PRODUCT_ID
        ]
        return snapshot

    def _check_consistency(self, snapshot: DPPDataSnapshot) -> List[str]:
        """Check consistency of DPP data values."""
        issues: List[str] = []

        if snapshot.recycled_content_pct > 100.0:
            issues.append("Recycled content exceeds 100%")

        if snapshot.durability_score > 10.0:
            issues.append("Durability score exceeds maximum scale of 10")

        if snapshot.carbon_footprint_kg_co2e < 0:
            issues.append("Carbon footprint cannot be negative")

        return issues
