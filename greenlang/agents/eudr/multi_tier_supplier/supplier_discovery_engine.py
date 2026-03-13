# -*- coding: utf-8 -*-
"""
Supplier Discovery Engine - AGENT-EUDR-008 Multi-Tier Supplier Tracker

Engine 1 of 8: Discovers sub-tier suppliers from multiple data sources
including supplier declarations, questionnaires, shipping documents,
certification databases, and ERP purchase orders. Implements recursive
discovery with configurable max depth, fuzzy deduplication, and
confidence scoring for all discovered supplier relationships.

Discovery Sources:
    - Tier 1 supplier declarations (self-reported sub-supplier lists)
    - Supplier questionnaires (structured and free-form responses)
    - Shipping documents (bills of lading, packing lists, certificates of origin)
    - Certification databases (RSPO, FSC, UTZ, Rainforest Alliance)
    - ERP purchase orders and goods receipts

Confidence Levels:
    - VERIFIED: Confirmed through multiple independent sources (score 0.90-1.0)
    - DECLARED: Reported by supplier in declaration/questionnaire (score 0.70-0.89)
    - INFERRED: Deduced from shipping docs, certification chains (score 0.40-0.69)
    - SUSPECTED: Pattern-based or indirect evidence only (score 0.10-0.39)

EUDR References:
    - Article 4: Due diligence on full supply chain
    - Article 9: Traceability information for every supplier
    - Article 10: Trader obligations to identify upstream operators

Zero-Hallucination Principle:
    All discovery logic is deterministic. Confidence scores use explicit
    rule-based formulas. No LLM calls in the scoring or discovery path.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum recursion depth for supplier chain discovery.
MAX_RECURSION_DEPTH: int = 15

#: Default batch size for batch discovery operations.
DEFAULT_BATCH_SIZE: int = 1000

#: Fuzzy match threshold for supplier name deduplication (0.0-1.0).
FUZZY_MATCH_THRESHOLD: float = 0.85

#: Minimum confidence score threshold for a relationship to be recorded.
MIN_CONFIDENCE_THRESHOLD: float = 0.10

#: Version string for this engine.
ENGINE_VERSION: str = "1.0.0"

#: Prometheus metric prefix.
METRIC_PREFIX: str = "gl_eudr_mst_"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DiscoverySource(str, Enum):
    """Source type for supplier discovery.

    Each source type has different baseline confidence and
    data quality characteristics.
    """

    DECLARATION = "declaration"
    QUESTIONNAIRE = "questionnaire"
    SHIPPING_DOC = "shipping_doc"
    CERTIFICATION = "certification"
    ERP = "erp"
    MANUAL = "manual"
    CROSS_REFERENCE = "cross_reference"


class ConfidenceLevel(str, Enum):
    """Confidence level for a discovered supplier relationship.

    Maps to score ranges used in regulatory reporting and
    risk assessment downstream.
    """

    VERIFIED = "verified"
    DECLARED = "declared"
    INFERRED = "inferred"
    SUSPECTED = "suspected"


class DiscoveryStatus(str, Enum):
    """Status of a discovery operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class EUDRCommodity(str, Enum):
    """EUDR regulated commodities (7 commodities per Article 1)."""

    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    SOYA = "soya"
    RUBBER = "rubber"
    CATTLE = "cattle"
    WOOD = "wood"


# ---------------------------------------------------------------------------
# Data Classes (local, independent of models.py)
# ---------------------------------------------------------------------------


@dataclass
class DiscoveredSupplier:
    """A supplier entity discovered from a data source.

    Attributes:
        supplier_id: Unique identifier for this supplier.
        name: Legal or trade name of the supplier.
        country_code: ISO 3166-1 alpha-2 country code.
        region: Administrative region or state.
        commodity_types: EUDR commodities handled by this supplier.
        registration_id: Legal entity registration number.
        tax_id: Tax identification number.
        duns_number: Dun & Bradstreet DUNS number.
        gps_latitude: GPS latitude of primary location.
        gps_longitude: GPS longitude of primary location.
        address: Physical address string.
        discovery_source: How this supplier was discovered.
        discovery_timestamp: When the supplier was discovered.
        raw_source_data: Original source data for provenance.
        metadata: Additional key-value metadata.
    """

    supplier_id: str = ""
    name: str = ""
    country_code: str = ""
    region: str = ""
    commodity_types: List[str] = field(default_factory=list)
    registration_id: str = ""
    tax_id: str = ""
    duns_number: str = ""
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    address: str = ""
    discovery_source: str = DiscoverySource.MANUAL.value
    discovery_timestamp: str = ""
    raw_source_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.supplier_id:
            self.supplier_id = str(uuid.uuid4())
        if not self.discovery_timestamp:
            self.discovery_timestamp = (
                datetime.now(timezone.utc).isoformat()
            )


@dataclass
class DiscoveredRelationship:
    """A discovered relationship between two supplier entities.

    Attributes:
        relationship_id: Unique identifier for this relationship.
        supplier_id: ID of the upstream supplier.
        buyer_id: ID of the downstream buyer.
        tier_level: Tier level of the upstream supplier relative to operator.
        commodity: Commodity traded in this relationship.
        confidence_score: Confidence score (0.0-1.0).
        confidence_level: Categorical confidence level.
        discovery_source: How this relationship was discovered.
        discovery_timestamp: When the relationship was discovered.
        volume_estimate_tonnes: Estimated annual volume in tonnes.
        evidence: Supporting evidence references.
        metadata: Additional key-value metadata.
    """

    relationship_id: str = ""
    supplier_id: str = ""
    buyer_id: str = ""
    tier_level: int = 0
    commodity: str = ""
    confidence_score: float = 0.0
    confidence_level: str = ConfidenceLevel.SUSPECTED.value
    discovery_source: str = DiscoverySource.MANUAL.value
    discovery_timestamp: str = ""
    volume_estimate_tonnes: Optional[float] = None
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.relationship_id:
            self.relationship_id = str(uuid.uuid4())
        if not self.discovery_timestamp:
            self.discovery_timestamp = (
                datetime.now(timezone.utc).isoformat()
            )


@dataclass
class DeclarationData:
    """Input data from a Tier 1 supplier declaration.

    Attributes:
        declaration_id: Unique declaration identifier.
        declaring_supplier_id: ID of the supplier making the declaration.
        declaring_supplier_name: Name of the declaring supplier.
        commodity: Commodity covered by this declaration.
        declared_suppliers: List of declared sub-tier suppliers.
        declaration_date: Date the declaration was made.
        declaration_type: Type of declaration (annual, ad-hoc, audit).
        metadata: Additional key-value metadata.
    """

    declaration_id: str = ""
    declaring_supplier_id: str = ""
    declaring_supplier_name: str = ""
    commodity: str = ""
    declared_suppliers: List[Dict[str, Any]] = field(default_factory=list)
    declaration_date: str = ""
    declaration_type: str = "annual"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionnaireData:
    """Input data from a supplier questionnaire response.

    Attributes:
        questionnaire_id: Unique questionnaire identifier.
        respondent_supplier_id: ID of the responding supplier.
        respondent_supplier_name: Name of the responding supplier.
        commodity: Commodity covered by this questionnaire.
        responses: List of question-answer pairs.
        submission_date: Date the questionnaire was submitted.
        questionnaire_version: Version of the questionnaire template.
        metadata: Additional key-value metadata.
    """

    questionnaire_id: str = ""
    respondent_supplier_id: str = ""
    respondent_supplier_name: str = ""
    commodity: str = ""
    responses: List[Dict[str, Any]] = field(default_factory=list)
    submission_date: str = ""
    questionnaire_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShippingData:
    """Input data from shipping documents (bills of lading, packing lists).

    Attributes:
        document_id: Unique document identifier.
        document_type: Type of shipping document.
        shipper_name: Name of the shipper entity.
        shipper_country: ISO country code of shipper.
        consignee_name: Name of the consignee entity.
        consignee_country: ISO country code of consignee.
        commodity: Commodity being shipped.
        volume_tonnes: Volume of commodity in tonnes.
        origin_port: Port of origin.
        destination_port: Port of destination.
        bill_of_lading_number: Bill of lading reference.
        shipment_date: Date of shipment.
        intermediate_parties: List of intermediate parties on the document.
        certificates: List of certificates attached to the shipment.
        metadata: Additional key-value metadata.
    """

    document_id: str = ""
    document_type: str = "bill_of_lading"
    shipper_name: str = ""
    shipper_country: str = ""
    consignee_name: str = ""
    consignee_country: str = ""
    commodity: str = ""
    volume_tonnes: float = 0.0
    origin_port: str = ""
    destination_port: str = ""
    bill_of_lading_number: str = ""
    shipment_date: str = ""
    intermediate_parties: List[Dict[str, Any]] = field(default_factory=list)
    certificates: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CertificationData:
    """Input data from certification database cross-references.

    Attributes:
        certification_id: Unique certification identifier.
        certification_type: Type of certification (FSC, RSPO, UTZ, etc.).
        certificate_number: Certificate reference number.
        holder_name: Name of the certificate holder.
        holder_country: ISO country code of holder.
        commodity: Commodity covered by certification.
        valid_from: Certification validity start date.
        valid_until: Certification validity end date.
        scope: Certification scope description.
        supply_chain_members: List of supply chain members under cert.
        certified_sites: List of certified sites/facilities.
        metadata: Additional key-value metadata.
    """

    certification_id: str = ""
    certification_type: str = ""
    certificate_number: str = ""
    holder_name: str = ""
    holder_country: str = ""
    commodity: str = ""
    valid_from: str = ""
    valid_until: str = ""
    scope: str = ""
    supply_chain_members: List[Dict[str, Any]] = field(default_factory=list)
    certified_sites: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ERPData:
    """Input data from ERP purchase orders and goods receipts.

    Attributes:
        erp_system: Source ERP system identifier.
        purchase_orders: List of purchase order records.
        goods_receipts: List of goods receipt records.
        vendor_master: Vendor master data records.
        material_documents: Material document records.
        metadata: Additional key-value metadata.
    """

    erp_system: str = ""
    purchase_orders: List[Dict[str, Any]] = field(default_factory=list)
    goods_receipts: List[Dict[str, Any]] = field(default_factory=list)
    vendor_master: List[Dict[str, Any]] = field(default_factory=list)
    material_documents: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Result of a supplier discovery operation.

    Attributes:
        result_id: Unique result identifier.
        status: Discovery operation status.
        discovered_suppliers: List of discovered supplier entities.
        discovered_relationships: List of discovered relationships.
        discovery_source: Source of the discovery.
        total_suppliers_found: Count of suppliers found.
        total_relationships_found: Count of relationships found.
        duplicates_detected: Count of duplicate suppliers merged.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        errors: List of errors encountered.
        warnings: List of warnings generated.
        timestamp: Result generation timestamp.
        metadata: Additional key-value metadata.
    """

    result_id: str = ""
    status: str = DiscoveryStatus.COMPLETED.value
    discovered_suppliers: List[DiscoveredSupplier] = field(
        default_factory=list
    )
    discovered_relationships: List[DiscoveredRelationship] = field(
        default_factory=list
    )
    discovery_source: str = ""
    total_suppliers_found: int = 0
    total_relationships_found: int = 0
    duplicates_detected: int = 0
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.result_id:
            self.result_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class BatchDiscoveryResult:
    """Result of a batch discovery operation.

    Attributes:
        batch_id: Unique batch identifier.
        status: Overall batch status.
        total_input_suppliers: Count of input supplier roots.
        total_discovered: Total suppliers discovered across all roots.
        total_relationships: Total relationships discovered.
        total_duplicates: Total duplicates merged.
        individual_results: Per-root discovery results.
        processing_time_ms: Total processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        errors: List of errors encountered.
        timestamp: Result generation timestamp.
    """

    batch_id: str = ""
    status: str = DiscoveryStatus.COMPLETED.value
    total_input_suppliers: int = 0
    total_discovered: int = 0
    total_relationships: int = 0
    total_duplicates: int = 0
    individual_results: List[DiscoveryResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    errors: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Confidence scoring reference data
# ---------------------------------------------------------------------------

#: Baseline confidence by discovery source.
_SOURCE_BASELINE_CONFIDENCE: Dict[str, float] = {
    DiscoverySource.DECLARATION.value: 0.75,
    DiscoverySource.QUESTIONNAIRE.value: 0.70,
    DiscoverySource.SHIPPING_DOC.value: 0.60,
    DiscoverySource.CERTIFICATION.value: 0.80,
    DiscoverySource.ERP.value: 0.85,
    DiscoverySource.MANUAL.value: 0.50,
    DiscoverySource.CROSS_REFERENCE.value: 0.65,
}

#: Confidence level thresholds (lower bound inclusive).
_CONFIDENCE_LEVEL_THRESHOLDS: List[Tuple[float, str]] = [
    (0.90, ConfidenceLevel.VERIFIED.value),
    (0.70, ConfidenceLevel.DECLARED.value),
    (0.40, ConfidenceLevel.INFERRED.value),
    (0.10, ConfidenceLevel.SUSPECTED.value),
]

#: Field presence bonus for confidence scoring.
_FIELD_CONFIDENCE_BONUS: Dict[str, float] = {
    "registration_id": 0.05,
    "tax_id": 0.05,
    "duns_number": 0.04,
    "gps_latitude": 0.03,
    "gps_longitude": 0.03,
    "country_code": 0.02,
    "address": 0.02,
    "commodity_types": 0.02,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _compute_provenance_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash for any serializable data.

    Args:
        data: Data to hash. Must be JSON-serializable or have a
            __dict__ attribute.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    try:
        if hasattr(data, "__dict__"):
            serialized = json.dumps(
                data.__dict__, sort_keys=True, default=str
            )
        else:
            serialized = json.dumps(data, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_name(name: str) -> str:
    """Normalize a supplier name for comparison.

    Converts to lowercase, strips whitespace, removes common
    legal suffixes (Ltd, LLC, Inc, GmbH, etc.), removes
    punctuation, and collapses multiple spaces.

    Args:
        name: Raw supplier name string.

    Returns:
        Normalized name string for comparison.
    """
    if not name:
        return ""
    normalized = name.lower().strip()
    # Remove common legal suffixes
    suffixes = [
        r"\bltd\.?\b", r"\bllc\.?\b", r"\binc\.?\b", r"\bcorp\.?\b",
        r"\bgmbh\.?\b", r"\bs\.?a\.?\b", r"\bs\.?a\.?r\.?l\.?\b",
        r"\bp\.?t\.?\b", r"\bco\.?\b", r"\bplc\.?\b", r"\bag\.?\b",
        r"\bb\.?v\.?\b", r"\bn\.?v\.?\b", r"\bpte\.?\b",
        r"\bpty\.?\b", r"\bsrl\.?\b", r"\bsl\.?\b",
    ]
    for suffix in suffixes:
        normalized = re.sub(suffix, "", normalized)
    # Remove punctuation
    normalized = re.sub(r"[^\w\s]", "", normalized)
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two normalized names using bigrams.

    Uses the Dice coefficient on character bigrams for efficient
    fuzzy matching without external dependencies.

    Args:
        name1: First normalized name.
        name2: Second normalized name.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not name1 or not name2:
        return 0.0
    if name1 == name2:
        return 1.0

    def _bigrams(s: str) -> List[str]:
        return [s[i:i + 2] for i in range(len(s) - 1)]

    bg1 = _bigrams(name1)
    bg2 = _bigrams(name2)
    if not bg1 or not bg2:
        return 0.0

    set1 = set(bg1)
    set2 = set(bg2)
    intersection_size = len(set1 & set2)
    return (2.0 * intersection_size) / (len(set1) + len(set2))


def _classify_confidence(score: float) -> str:
    """Classify a numeric confidence score into a confidence level.

    Args:
        score: Confidence score (0.0-1.0).

    Returns:
        ConfidenceLevel value string.
    """
    for threshold, level in _CONFIDENCE_LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return ConfidenceLevel.SUSPECTED.value


# ---------------------------------------------------------------------------
# SupplierDiscoveryEngine
# ---------------------------------------------------------------------------


class SupplierDiscoveryEngine:
    """Engine 1: Discovers sub-tier suppliers from multiple data sources.

    Implements EUDR Article 4 and Article 9 requirements for full supply
    chain traceability by discovering suppliers from declarations,
    questionnaires, shipping documents, certification databases, and
    ERP purchase orders.

    All discovery operations are deterministic and produce SHA-256
    provenance hashes for complete audit trails. Confidence scoring
    follows explicit rule-based formulas with no LLM involvement.

    Attributes:
        max_recursion_depth: Maximum depth for recursive discovery.
        fuzzy_threshold: Similarity threshold for deduplication.
        min_confidence: Minimum confidence to record a relationship.
        batch_size: Default batch processing chunk size.
        _discovery_count: Running count of discoveries for metrics.
        _relationship_count: Running count of relationships for metrics.

    Example:
        >>> engine = SupplierDiscoveryEngine()
        >>> decl = DeclarationData(
        ...     declaring_supplier_id="SUP-001",
        ...     commodity="cocoa",
        ...     declared_suppliers=[{"name": "Coop A", "country": "GH"}],
        ... )
        >>> result = engine.discover_from_declaration(decl)
        >>> assert result.total_suppliers_found >= 1
    """

    def __init__(
        self,
        max_recursion_depth: int = MAX_RECURSION_DEPTH,
        fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialize SupplierDiscoveryEngine.

        Args:
            max_recursion_depth: Maximum depth for recursive discovery.
                Must be between 1 and 50.
            fuzzy_threshold: Similarity threshold for deduplication.
                Must be between 0.0 and 1.0.
            min_confidence: Minimum confidence to record a relationship.
                Must be between 0.0 and 1.0.
            batch_size: Default batch processing chunk size. Must be >= 1.

        Raises:
            ValueError: If any parameter is outside valid range.
        """
        errors: List[str] = []
        if not 1 <= max_recursion_depth <= 50:
            errors.append(
                f"max_recursion_depth must be in [1, 50], "
                f"got {max_recursion_depth}"
            )
        if not 0.0 <= fuzzy_threshold <= 1.0:
            errors.append(
                f"fuzzy_threshold must be in [0.0, 1.0], "
                f"got {fuzzy_threshold}"
            )
        if not 0.0 <= min_confidence <= 1.0:
            errors.append(
                f"min_confidence must be in [0.0, 1.0], "
                f"got {min_confidence}"
            )
        if batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {batch_size}")
        if errors:
            raise ValueError(
                "SupplierDiscoveryEngine init failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        self.max_recursion_depth = max_recursion_depth
        self.fuzzy_threshold = fuzzy_threshold
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self._discovery_count: int = 0
        self._relationship_count: int = 0

        logger.info(
            "SupplierDiscoveryEngine initialized: "
            "max_depth=%d, fuzzy_threshold=%.2f, "
            "min_confidence=%.2f, batch_size=%d",
            self.max_recursion_depth,
            self.fuzzy_threshold,
            self.min_confidence,
            self.batch_size,
        )

    # ------------------------------------------------------------------
    # Discovery: from declaration
    # ------------------------------------------------------------------

    def discover_from_declaration(
        self, declaration_data: DeclarationData
    ) -> DiscoveryResult:
        """Parse Tier 1 supplier declarations to find sub-tier suppliers.

        Extracts supplier entities and relationships from structured
        declaration data where a Tier 1 supplier self-reports their
        upstream supply chain.

        Args:
            declaration_data: Parsed declaration data containing
                the declaring supplier and their declared sub-suppliers.

        Returns:
            DiscoveryResult with discovered suppliers and relationships.

        Raises:
            ValueError: If declaration_data is missing required fields.
        """
        start_time = time.monotonic()
        logger.info(
            "Starting declaration discovery: declaration_id=%s, "
            "declaring_supplier=%s, commodity=%s, "
            "declared_count=%d",
            declaration_data.declaration_id,
            declaration_data.declaring_supplier_id,
            declaration_data.commodity,
            len(declaration_data.declared_suppliers),
        )

        discovered_suppliers: List[DiscoveredSupplier] = []
        discovered_relationships: List[DiscoveredRelationship] = []
        errors: List[str] = []
        warnings: List[str] = []

        if not declaration_data.declaring_supplier_id:
            errors.append("declaring_supplier_id is required")
            return self._build_result(
                source=DiscoverySource.DECLARATION.value,
                suppliers=discovered_suppliers,
                relationships=discovered_relationships,
                errors=errors,
                warnings=warnings,
                start_time=start_time,
            )

        for idx, declared in enumerate(
            declaration_data.declared_suppliers
        ):
            try:
                supplier = self._extract_supplier_from_declaration(
                    declared, declaration_data, idx
                )
                if supplier is not None:
                    discovered_suppliers.append(supplier)

                    relationship = self._create_relationship(
                        supplier_id=supplier.supplier_id,
                        buyer_id=declaration_data.declaring_supplier_id,
                        commodity=declaration_data.commodity,
                        source=DiscoverySource.DECLARATION.value,
                        tier_level=self._extract_tier_level(declared),
                        volume=self._extract_volume(declared),
                        evidence=[
                            f"declaration:{declaration_data.declaration_id}",
                            f"index:{idx}",
                        ],
                    )
                    discovered_relationships.append(relationship)
                else:
                    warnings.append(
                        f"Could not extract supplier from declaration "
                        f"entry {idx}: insufficient data"
                    )
            except Exception as exc:
                error_msg = (
                    f"Error processing declaration entry {idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_msg)
                logger.warning(error_msg)

        result = self._build_result(
            source=DiscoverySource.DECLARATION.value,
            suppliers=discovered_suppliers,
            relationships=discovered_relationships,
            errors=errors,
            warnings=warnings,
            start_time=start_time,
        )

        self._discovery_count += result.total_suppliers_found
        self._relationship_count += result.total_relationships_found

        logger.info(
            "Declaration discovery completed: result_id=%s, "
            "suppliers=%d, relationships=%d, errors=%d, "
            "duration_ms=%.2f",
            result.result_id,
            result.total_suppliers_found,
            result.total_relationships_found,
            len(result.errors),
            result.processing_time_ms,
        )
        return result

    def _extract_supplier_from_declaration(
        self,
        declared: Dict[str, Any],
        declaration_data: DeclarationData,
        index: int,
    ) -> Optional[DiscoveredSupplier]:
        """Extract a supplier entity from a single declaration entry.

        Args:
            declared: Dictionary with declared supplier data.
            declaration_data: Parent declaration context.
            index: Index within the declaration list.

        Returns:
            DiscoveredSupplier if sufficient data present, None otherwise.
        """
        name = str(declared.get("name", "")).strip()
        if not name:
            return None

        supplier = DiscoveredSupplier(
            name=name,
            country_code=str(declared.get("country", "")).strip().upper(),
            region=str(declared.get("region", "")).strip(),
            commodity_types=self._extract_commodity_list(
                declared, declaration_data.commodity
            ),
            registration_id=str(
                declared.get("registration_id", "")
            ).strip(),
            tax_id=str(declared.get("tax_id", "")).strip(),
            duns_number=str(declared.get("duns_number", "")).strip(),
            gps_latitude=self._safe_float(declared.get("latitude")),
            gps_longitude=self._safe_float(declared.get("longitude")),
            address=str(declared.get("address", "")).strip(),
            discovery_source=DiscoverySource.DECLARATION.value,
            raw_source_data={
                "declaration_id": declaration_data.declaration_id,
                "entry_index": index,
                "entry_data": declared,
            },
        )
        return supplier

    # ------------------------------------------------------------------
    # Discovery: from questionnaire
    # ------------------------------------------------------------------

    def discover_from_questionnaire(
        self, questionnaire_data: QuestionnaireData
    ) -> DiscoveryResult:
        """Extract supplier relationships from questionnaire responses.

        Parses structured question-answer pairs to identify supplier
        names, locations, commodities, and supply chain relationships.
        Questions about sourcing patterns, upstream suppliers, and
        processing facilities are key discovery targets.

        Args:
            questionnaire_data: Parsed questionnaire response data.

        Returns:
            DiscoveryResult with discovered suppliers and relationships.

        Raises:
            ValueError: If questionnaire_data is missing required fields.
        """
        start_time = time.monotonic()
        logger.info(
            "Starting questionnaire discovery: questionnaire_id=%s, "
            "respondent=%s, commodity=%s, responses=%d",
            questionnaire_data.questionnaire_id,
            questionnaire_data.respondent_supplier_id,
            questionnaire_data.commodity,
            len(questionnaire_data.responses),
        )

        discovered_suppliers: List[DiscoveredSupplier] = []
        discovered_relationships: List[DiscoveredRelationship] = []
        errors: List[str] = []
        warnings: List[str] = []

        if not questionnaire_data.respondent_supplier_id:
            errors.append("respondent_supplier_id is required")
            return self._build_result(
                source=DiscoverySource.QUESTIONNAIRE.value,
                suppliers=discovered_suppliers,
                relationships=discovered_relationships,
                errors=errors,
                warnings=warnings,
                start_time=start_time,
            )

        # Categorize responses by type for targeted extraction
        supplier_responses = self._filter_supplier_responses(
            questionnaire_data.responses
        )

        for idx, response in enumerate(supplier_responses):
            try:
                supplier = self._extract_supplier_from_response(
                    response, questionnaire_data, idx
                )
                if supplier is not None:
                    discovered_suppliers.append(supplier)

                    relationship = self._create_relationship(
                        supplier_id=supplier.supplier_id,
                        buyer_id=(
                            questionnaire_data.respondent_supplier_id
                        ),
                        commodity=questionnaire_data.commodity,
                        source=DiscoverySource.QUESTIONNAIRE.value,
                        tier_level=self._extract_tier_from_response(
                            response
                        ),
                        volume=self._extract_volume_from_response(
                            response
                        ),
                        evidence=[
                            f"questionnaire:"
                            f"{questionnaire_data.questionnaire_id}",
                            f"response_index:{idx}",
                        ],
                    )
                    discovered_relationships.append(relationship)
            except Exception as exc:
                error_msg = (
                    f"Error processing questionnaire response {idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_msg)
                logger.warning(error_msg)

        result = self._build_result(
            source=DiscoverySource.QUESTIONNAIRE.value,
            suppliers=discovered_suppliers,
            relationships=discovered_relationships,
            errors=errors,
            warnings=warnings,
            start_time=start_time,
        )

        self._discovery_count += result.total_suppliers_found
        self._relationship_count += result.total_relationships_found

        logger.info(
            "Questionnaire discovery completed: result_id=%s, "
            "suppliers=%d, relationships=%d, duration_ms=%.2f",
            result.result_id,
            result.total_suppliers_found,
            result.total_relationships_found,
            result.processing_time_ms,
        )
        return result

    def _filter_supplier_responses(
        self, responses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter questionnaire responses to those containing supplier data.

        Identifies responses that reference supplier names, sourcing
        locations, upstream partners, processing facilities, or
        sub-supplier lists.

        Args:
            responses: All questionnaire responses.

        Returns:
            Filtered list of responses likely to contain supplier info.
        """
        supplier_keywords = {
            "supplier", "source", "upstream", "processor", "mill",
            "cooperative", "aggregator", "trader", "exporter",
            "farmer", "grower", "plantation", "ranch", "feedlot",
            "sawmill", "refinery", "factory", "dealer",
            "sub-supplier", "sub_supplier", "subsupplier",
        }
        filtered: List[Dict[str, Any]] = []
        for response in responses:
            question = str(response.get("question", "")).lower()
            answer = str(response.get("answer", "")).lower()
            category = str(response.get("category", "")).lower()

            if any(
                kw in question or kw in answer or kw in category
                for kw in supplier_keywords
            ):
                filtered.append(response)

            # Also include if answer contains structured supplier data
            if isinstance(response.get("answer"), (list, dict)):
                filtered.append(response)

        return filtered

    def _extract_supplier_from_response(
        self,
        response: Dict[str, Any],
        questionnaire_data: QuestionnaireData,
        index: int,
    ) -> Optional[DiscoveredSupplier]:
        """Extract supplier entity from a questionnaire response.

        Args:
            response: Single question-answer pair.
            questionnaire_data: Parent questionnaire context.
            index: Index of this response within filtered responses.

        Returns:
            DiscoveredSupplier if sufficient data present, None otherwise.
        """
        answer = response.get("answer", "")

        # Handle structured answer (dict with supplier fields)
        if isinstance(answer, dict):
            name = str(answer.get("name", "")).strip()
            if not name:
                return None
            return DiscoveredSupplier(
                name=name,
                country_code=str(
                    answer.get("country", "")
                ).strip().upper(),
                region=str(answer.get("region", "")).strip(),
                commodity_types=self._extract_commodity_list(
                    answer, questionnaire_data.commodity
                ),
                registration_id=str(
                    answer.get("registration_id", "")
                ).strip(),
                gps_latitude=self._safe_float(answer.get("latitude")),
                gps_longitude=self._safe_float(answer.get("longitude")),
                address=str(answer.get("address", "")).strip(),
                discovery_source=DiscoverySource.QUESTIONNAIRE.value,
                raw_source_data={
                    "questionnaire_id": (
                        questionnaire_data.questionnaire_id
                    ),
                    "response_index": index,
                    "response": response,
                },
            )

        # Handle list of supplier names
        if isinstance(answer, list):
            # Return the first valid supplier; caller handles lists
            for item in answer:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                elif isinstance(item, str):
                    name = item.strip()
                else:
                    continue
                if name:
                    return DiscoveredSupplier(
                        name=name,
                        commodity_types=[questionnaire_data.commodity]
                        if questionnaire_data.commodity
                        else [],
                        discovery_source=(
                            DiscoverySource.QUESTIONNAIRE.value
                        ),
                        raw_source_data={
                            "questionnaire_id": (
                                questionnaire_data.questionnaire_id
                            ),
                            "response_index": index,
                        },
                    )
            return None

        # Handle free-text answer - extract supplier name if present
        text = str(answer).strip()
        if text and len(text) >= 2:
            return DiscoveredSupplier(
                name=text,
                commodity_types=[questionnaire_data.commodity]
                if questionnaire_data.commodity
                else [],
                discovery_source=DiscoverySource.QUESTIONNAIRE.value,
                raw_source_data={
                    "questionnaire_id": (
                        questionnaire_data.questionnaire_id
                    ),
                    "response_index": index,
                },
            )
        return None

    def _extract_tier_from_response(
        self, response: Dict[str, Any]
    ) -> int:
        """Extract tier level from a questionnaire response.

        Args:
            response: Question-answer pair that may include tier info.

        Returns:
            Tier level integer, defaults to 2 if not specified.
        """
        answer = response.get("answer", {})
        if isinstance(answer, dict):
            tier = answer.get("tier", answer.get("tier_level"))
            if tier is not None:
                try:
                    return max(1, int(tier))
                except (ValueError, TypeError):
                    pass
        question = str(response.get("question", "")).lower()
        # Heuristic: if question mentions Tier N, extract N+1
        tier_match = re.search(r"tier\s*(\d+)", question)
        if tier_match:
            return int(tier_match.group(1)) + 1
        return 2

    def _extract_volume_from_response(
        self, response: Dict[str, Any]
    ) -> Optional[float]:
        """Extract volume estimate from a questionnaire response.

        Args:
            response: Question-answer pair.

        Returns:
            Volume in tonnes if available, None otherwise.
        """
        answer = response.get("answer", {})
        if isinstance(answer, dict):
            volume = answer.get(
                "volume",
                answer.get("volume_tonnes", answer.get("quantity")),
            )
            return self._safe_float(volume)
        return None

    # ------------------------------------------------------------------
    # Discovery: from shipping documents
    # ------------------------------------------------------------------

    def discover_from_shipping_docs(
        self, shipping_data: ShippingData
    ) -> DiscoveryResult:
        """Extract suppliers from bills of lading and packing lists.

        Analyzes shipping documents to identify shipper, consignee,
        and intermediate parties as supply chain participants.
        Extracts origin and routing information to establish
        geographic context.

        Args:
            shipping_data: Parsed shipping document data.

        Returns:
            DiscoveryResult with discovered suppliers and relationships.
        """
        start_time = time.monotonic()
        logger.info(
            "Starting shipping doc discovery: document_id=%s, "
            "type=%s, shipper=%s, consignee=%s, commodity=%s",
            shipping_data.document_id,
            shipping_data.document_type,
            shipping_data.shipper_name,
            shipping_data.consignee_name,
            shipping_data.commodity,
        )

        discovered_suppliers: List[DiscoveredSupplier] = []
        discovered_relationships: List[DiscoveredRelationship] = []
        errors: List[str] = []
        warnings: List[str] = []

        # Extract shipper as a supplier
        if shipping_data.shipper_name:
            shipper = DiscoveredSupplier(
                name=shipping_data.shipper_name,
                country_code=shipping_data.shipper_country.upper()
                if shipping_data.shipper_country
                else "",
                commodity_types=[shipping_data.commodity]
                if shipping_data.commodity
                else [],
                discovery_source=DiscoverySource.SHIPPING_DOC.value,
                raw_source_data={
                    "document_id": shipping_data.document_id,
                    "role": "shipper",
                    "bl_number": shipping_data.bill_of_lading_number,
                },
            )
            discovered_suppliers.append(shipper)

            # If consignee is known, create relationship
            if shipping_data.consignee_name:
                consignee = DiscoveredSupplier(
                    name=shipping_data.consignee_name,
                    country_code=(
                        shipping_data.consignee_country.upper()
                        if shipping_data.consignee_country
                        else ""
                    ),
                    commodity_types=[shipping_data.commodity]
                    if shipping_data.commodity
                    else [],
                    discovery_source=(
                        DiscoverySource.SHIPPING_DOC.value
                    ),
                    raw_source_data={
                        "document_id": shipping_data.document_id,
                        "role": "consignee",
                    },
                )
                discovered_suppliers.append(consignee)

                relationship = self._create_relationship(
                    supplier_id=shipper.supplier_id,
                    buyer_id=consignee.supplier_id,
                    commodity=shipping_data.commodity,
                    source=DiscoverySource.SHIPPING_DOC.value,
                    tier_level=1,
                    volume=shipping_data.volume_tonnes
                    if shipping_data.volume_tonnes > 0
                    else None,
                    evidence=[
                        f"shipping_doc:{shipping_data.document_id}",
                        f"bl:{shipping_data.bill_of_lading_number}",
                    ],
                )
                discovered_relationships.append(relationship)

        # Extract intermediate parties
        for idx, party in enumerate(
            shipping_data.intermediate_parties
        ):
            try:
                party_name = str(party.get("name", "")).strip()
                if not party_name:
                    continue
                party_supplier = DiscoveredSupplier(
                    name=party_name,
                    country_code=str(
                        party.get("country", "")
                    ).strip().upper(),
                    commodity_types=[shipping_data.commodity]
                    if shipping_data.commodity
                    else [],
                    discovery_source=(
                        DiscoverySource.SHIPPING_DOC.value
                    ),
                    raw_source_data={
                        "document_id": shipping_data.document_id,
                        "role": str(
                            party.get("role", "intermediate")
                        ),
                        "party_index": idx,
                    },
                    metadata={
                        "party_role": str(
                            party.get("role", "intermediate")
                        ),
                    },
                )
                discovered_suppliers.append(party_supplier)
            except Exception as exc:
                error_msg = (
                    f"Error processing intermediate party {idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_msg)
                logger.warning(error_msg)

        # Extract from attached certificates
        for idx, cert in enumerate(shipping_data.certificates):
            try:
                cert_holder = str(
                    cert.get("holder_name", cert.get("holder", ""))
                ).strip()
                if cert_holder and cert_holder not in [
                    s.name for s in discovered_suppliers
                ]:
                    cert_supplier = DiscoveredSupplier(
                        name=cert_holder,
                        country_code=str(
                            cert.get("country", "")
                        ).strip().upper(),
                        commodity_types=[shipping_data.commodity]
                        if shipping_data.commodity
                        else [],
                        discovery_source=(
                            DiscoverySource.SHIPPING_DOC.value
                        ),
                        raw_source_data={
                            "document_id": shipping_data.document_id,
                            "cert_type": str(
                                cert.get("type", "unknown")
                            ),
                            "cert_number": str(
                                cert.get("number", "")
                            ),
                        },
                    )
                    discovered_suppliers.append(cert_supplier)
            except Exception as exc:
                error_msg = (
                    f"Error processing certificate {idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_msg)
                logger.warning(error_msg)

        result = self._build_result(
            source=DiscoverySource.SHIPPING_DOC.value,
            suppliers=discovered_suppliers,
            relationships=discovered_relationships,
            errors=errors,
            warnings=warnings,
            start_time=start_time,
        )

        self._discovery_count += result.total_suppliers_found
        self._relationship_count += result.total_relationships_found

        logger.info(
            "Shipping doc discovery completed: result_id=%s, "
            "suppliers=%d, relationships=%d, duration_ms=%.2f",
            result.result_id,
            result.total_suppliers_found,
            result.total_relationships_found,
            result.processing_time_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Discovery: from certification
    # ------------------------------------------------------------------

    def discover_from_certification(
        self, cert_data: CertificationData
    ) -> DiscoveryResult:
        """Cross-reference certification databases to discover suppliers.

        Extracts supply chain members and certified sites from
        certification records (FSC, RSPO, UTZ, Rainforest Alliance)
        to identify upstream suppliers.

        Args:
            cert_data: Parsed certification database record.

        Returns:
            DiscoveryResult with discovered suppliers and relationships.
        """
        start_time = time.monotonic()
        logger.info(
            "Starting certification discovery: cert_id=%s, "
            "type=%s, holder=%s, commodity=%s, "
            "members=%d, sites=%d",
            cert_data.certification_id,
            cert_data.certification_type,
            cert_data.holder_name,
            cert_data.commodity,
            len(cert_data.supply_chain_members),
            len(cert_data.certified_sites),
        )

        discovered_suppliers: List[DiscoveredSupplier] = []
        discovered_relationships: List[DiscoveredRelationship] = []
        errors: List[str] = []
        warnings: List[str] = []

        # Create supplier for certification holder
        holder_supplier: Optional[DiscoveredSupplier] = None
        if cert_data.holder_name:
            holder_supplier = DiscoveredSupplier(
                name=cert_data.holder_name,
                country_code=cert_data.holder_country.upper()
                if cert_data.holder_country
                else "",
                commodity_types=[cert_data.commodity]
                if cert_data.commodity
                else [],
                discovery_source=DiscoverySource.CERTIFICATION.value,
                raw_source_data={
                    "certification_id": cert_data.certification_id,
                    "cert_type": cert_data.certification_type,
                    "cert_number": cert_data.certificate_number,
                    "role": "holder",
                },
                metadata={
                    "certification_type": cert_data.certification_type,
                    "certificate_number": cert_data.certificate_number,
                    "valid_from": cert_data.valid_from,
                    "valid_until": cert_data.valid_until,
                },
            )
            discovered_suppliers.append(holder_supplier)

        # Extract supply chain members
        for idx, member in enumerate(cert_data.supply_chain_members):
            try:
                member_name = str(
                    member.get("name", "")
                ).strip()
                if not member_name:
                    continue

                member_supplier = DiscoveredSupplier(
                    name=member_name,
                    country_code=str(
                        member.get("country", "")
                    ).strip().upper(),
                    region=str(member.get("region", "")).strip(),
                    commodity_types=[cert_data.commodity]
                    if cert_data.commodity
                    else [],
                    registration_id=str(
                        member.get("registration_id", "")
                    ).strip(),
                    gps_latitude=self._safe_float(
                        member.get("latitude")
                    ),
                    gps_longitude=self._safe_float(
                        member.get("longitude")
                    ),
                    discovery_source=(
                        DiscoverySource.CERTIFICATION.value
                    ),
                    raw_source_data={
                        "certification_id": (
                            cert_data.certification_id
                        ),
                        "member_index": idx,
                        "member_role": str(
                            member.get("role", "member")
                        ),
                    },
                    metadata={
                        "certification_type": (
                            cert_data.certification_type
                        ),
                        "member_role": str(
                            member.get("role", "member")
                        ),
                    },
                )
                discovered_suppliers.append(member_supplier)

                # Create relationship to holder if holder exists
                if holder_supplier is not None:
                    relationship = self._create_relationship(
                        supplier_id=member_supplier.supplier_id,
                        buyer_id=holder_supplier.supplier_id,
                        commodity=cert_data.commodity,
                        source=DiscoverySource.CERTIFICATION.value,
                        tier_level=self._extract_tier_from_role(
                            str(member.get("role", "member"))
                        ),
                        evidence=[
                            f"certification:"
                            f"{cert_data.certification_id}",
                            f"member:{member_name}",
                        ],
                    )
                    discovered_relationships.append(relationship)
            except Exception as exc:
                error_msg = (
                    f"Error processing cert member {idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_msg)
                logger.warning(error_msg)

        # Extract certified sites as suppliers
        for idx, site in enumerate(cert_data.certified_sites):
            try:
                site_name = str(site.get("name", "")).strip()
                if not site_name:
                    continue

                site_supplier = DiscoveredSupplier(
                    name=site_name,
                    country_code=str(
                        site.get("country", "")
                    ).strip().upper(),
                    gps_latitude=self._safe_float(
                        site.get("latitude")
                    ),
                    gps_longitude=self._safe_float(
                        site.get("longitude")
                    ),
                    address=str(site.get("address", "")).strip(),
                    commodity_types=[cert_data.commodity]
                    if cert_data.commodity
                    else [],
                    discovery_source=(
                        DiscoverySource.CERTIFICATION.value
                    ),
                    raw_source_data={
                        "certification_id": (
                            cert_data.certification_id
                        ),
                        "site_index": idx,
                        "site_type": str(
                            site.get("type", "facility")
                        ),
                    },
                    metadata={
                        "site_type": str(
                            site.get("type", "facility")
                        ),
                        "certification_type": (
                            cert_data.certification_type
                        ),
                    },
                )
                discovered_suppliers.append(site_supplier)
            except Exception as exc:
                error_msg = (
                    f"Error processing cert site {idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_msg)
                logger.warning(error_msg)

        result = self._build_result(
            source=DiscoverySource.CERTIFICATION.value,
            suppliers=discovered_suppliers,
            relationships=discovered_relationships,
            errors=errors,
            warnings=warnings,
            start_time=start_time,
        )

        self._discovery_count += result.total_suppliers_found
        self._relationship_count += result.total_relationships_found

        logger.info(
            "Certification discovery completed: result_id=%s, "
            "suppliers=%d, relationships=%d, duration_ms=%.2f",
            result.result_id,
            result.total_suppliers_found,
            result.total_relationships_found,
            result.processing_time_ms,
        )
        return result

    def _extract_tier_from_role(self, role: str) -> int:
        """Map a certification member role to a tier level.

        Args:
            role: Member role string (e.g., 'processor', 'farmer').

        Returns:
            Tier level integer.
        """
        role_tier_map: Dict[str, int] = {
            "trader": 1,
            "exporter": 1,
            "processor": 2,
            "refinery": 2,
            "mill": 2,
            "aggregator": 3,
            "collector": 3,
            "cooperative": 3,
            "dealer": 3,
            "farmer": 4,
            "grower": 4,
            "plantation": 4,
            "smallholder": 5,
            "ranch": 4,
            "sawmill": 2,
            "member": 2,
        }
        return role_tier_map.get(role.lower().strip(), 2)

    # ------------------------------------------------------------------
    # Discovery: from ERP
    # ------------------------------------------------------------------

    def discover_from_erp(self, erp_data: ERPData) -> DiscoveryResult:
        """Extract suppliers from ERP purchase orders and vendor master.

        Analyzes purchase orders, goods receipts, and vendor master
        data to identify supplier entities and their relationships.
        ERP data typically provides the highest baseline confidence
        due to financial verification.

        Args:
            erp_data: Parsed ERP data including POs, GRs, and
                vendor master records.

        Returns:
            DiscoveryResult with discovered suppliers and relationships.
        """
        start_time = time.monotonic()
        logger.info(
            "Starting ERP discovery: system=%s, POs=%d, GRs=%d, "
            "vendors=%d, material_docs=%d",
            erp_data.erp_system,
            len(erp_data.purchase_orders),
            len(erp_data.goods_receipts),
            len(erp_data.vendor_master),
            len(erp_data.material_documents),
        )

        discovered_suppliers: List[DiscoveredSupplier] = []
        discovered_relationships: List[DiscoveredRelationship] = []
        errors: List[str] = []
        warnings: List[str] = []
        seen_vendor_ids: Set[str] = set()

        # Extract from vendor master data first (richest data)
        for idx, vendor in enumerate(erp_data.vendor_master):
            try:
                supplier = self._extract_supplier_from_vendor(
                    vendor, erp_data.erp_system, idx
                )
                if supplier is not None:
                    vendor_id = str(
                        vendor.get(
                            "vendor_id",
                            vendor.get("supplier_number", ""),
                        )
                    ).strip()
                    if vendor_id:
                        seen_vendor_ids.add(vendor_id)
                    discovered_suppliers.append(supplier)
            except Exception as exc:
                error_msg = (
                    f"Error processing vendor {idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_msg)
                logger.warning(error_msg)

        # Extract from purchase orders (relationships)
        for idx, po in enumerate(erp_data.purchase_orders):
            try:
                po_supplier = self._extract_supplier_from_po(
                    po, erp_data.erp_system, idx
                )
                if po_supplier is None:
                    continue

                vendor_id = str(
                    po.get("vendor_id", po.get("supplier_number", ""))
                ).strip()
                if vendor_id and vendor_id not in seen_vendor_ids:
                    discovered_suppliers.append(po_supplier)
                    seen_vendor_ids.add(vendor_id)

                # Create relationship: supplier -> buyer (our org)
                buyer_id = str(
                    po.get(
                        "buying_org",
                        po.get("company_code", "OPERATOR"),
                    )
                ).strip()
                commodity = str(
                    po.get(
                        "commodity",
                        po.get("material_group", ""),
                    )
                ).strip()
                volume = self._safe_float(
                    po.get("quantity", po.get("volume"))
                )

                relationship = self._create_relationship(
                    supplier_id=po_supplier.supplier_id,
                    buyer_id=buyer_id,
                    commodity=commodity,
                    source=DiscoverySource.ERP.value,
                    tier_level=1,
                    volume=volume,
                    evidence=[
                        f"erp:{erp_data.erp_system}",
                        f"po:{po.get('po_number', 'unknown')}",
                    ],
                )
                discovered_relationships.append(relationship)
            except Exception as exc:
                error_msg = (
                    f"Error processing PO {idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_msg)
                logger.warning(error_msg)

        # Extract from goods receipts for volume validation
        for idx, gr in enumerate(erp_data.goods_receipts):
            try:
                vendor_id = str(
                    gr.get("vendor_id", gr.get("supplier_number", ""))
                ).strip()
                if vendor_id and vendor_id not in seen_vendor_ids:
                    gr_supplier = self._extract_supplier_from_gr(
                        gr, erp_data.erp_system, idx
                    )
                    if gr_supplier is not None:
                        discovered_suppliers.append(gr_supplier)
                        seen_vendor_ids.add(vendor_id)
            except Exception as exc:
                error_msg = (
                    f"Error processing GR {idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_msg)
                logger.warning(error_msg)

        result = self._build_result(
            source=DiscoverySource.ERP.value,
            suppliers=discovered_suppliers,
            relationships=discovered_relationships,
            errors=errors,
            warnings=warnings,
            start_time=start_time,
        )

        self._discovery_count += result.total_suppliers_found
        self._relationship_count += result.total_relationships_found

        logger.info(
            "ERP discovery completed: result_id=%s, "
            "suppliers=%d, relationships=%d, duration_ms=%.2f",
            result.result_id,
            result.total_suppliers_found,
            result.total_relationships_found,
            result.processing_time_ms,
        )
        return result

    def _extract_supplier_from_vendor(
        self,
        vendor: Dict[str, Any],
        erp_system: str,
        index: int,
    ) -> Optional[DiscoveredSupplier]:
        """Extract supplier from ERP vendor master record.

        Args:
            vendor: Vendor master record dictionary.
            erp_system: Source ERP system identifier.
            index: Index within vendor list.

        Returns:
            DiscoveredSupplier if name is present, None otherwise.
        """
        name = str(
            vendor.get("name", vendor.get("vendor_name", ""))
        ).strip()
        if not name:
            return None

        return DiscoveredSupplier(
            name=name,
            country_code=str(
                vendor.get("country", vendor.get("country_code", ""))
            ).strip().upper(),
            region=str(
                vendor.get("region", vendor.get("state", ""))
            ).strip(),
            commodity_types=self._extract_commodity_list(vendor, ""),
            registration_id=str(
                vendor.get("registration_id", vendor.get("reg_no", ""))
            ).strip(),
            tax_id=str(
                vendor.get("tax_id", vendor.get("vat_number", ""))
            ).strip(),
            duns_number=str(
                vendor.get("duns_number", vendor.get("duns", ""))
            ).strip(),
            address=str(
                vendor.get("address", vendor.get("street", ""))
            ).strip(),
            discovery_source=DiscoverySource.ERP.value,
            raw_source_data={
                "erp_system": erp_system,
                "vendor_index": index,
                "vendor_id": str(
                    vendor.get(
                        "vendor_id",
                        vendor.get("supplier_number", ""),
                    )
                ),
            },
        )

    def _extract_supplier_from_po(
        self,
        po: Dict[str, Any],
        erp_system: str,
        index: int,
    ) -> Optional[DiscoveredSupplier]:
        """Extract supplier from ERP purchase order.

        Args:
            po: Purchase order record dictionary.
            erp_system: Source ERP system identifier.
            index: Index within PO list.

        Returns:
            DiscoveredSupplier if vendor name present, None otherwise.
        """
        name = str(
            po.get("vendor_name", po.get("supplier_name", ""))
        ).strip()
        if not name:
            return None

        return DiscoveredSupplier(
            name=name,
            country_code=str(
                po.get("vendor_country", po.get("country", ""))
            ).strip().upper(),
            commodity_types=[po.get("commodity", "")]
            if po.get("commodity")
            else [],
            discovery_source=DiscoverySource.ERP.value,
            raw_source_data={
                "erp_system": erp_system,
                "po_index": index,
                "po_number": str(po.get("po_number", "")),
            },
        )

    def _extract_supplier_from_gr(
        self,
        gr: Dict[str, Any],
        erp_system: str,
        index: int,
    ) -> Optional[DiscoveredSupplier]:
        """Extract supplier from ERP goods receipt.

        Args:
            gr: Goods receipt record dictionary.
            erp_system: Source ERP system identifier.
            index: Index within GR list.

        Returns:
            DiscoveredSupplier if vendor name present, None otherwise.
        """
        name = str(
            gr.get("vendor_name", gr.get("supplier_name", ""))
        ).strip()
        if not name:
            return None

        return DiscoveredSupplier(
            name=name,
            country_code=str(
                gr.get("vendor_country", gr.get("country", ""))
            ).strip().upper(),
            discovery_source=DiscoverySource.ERP.value,
            raw_source_data={
                "erp_system": erp_system,
                "gr_index": index,
                "gr_number": str(gr.get("gr_number", "")),
            },
        )

    # ------------------------------------------------------------------
    # Recursive discovery
    # ------------------------------------------------------------------

    def recursive_discover(
        self,
        root_supplier: DiscoveredSupplier,
        known_relationships: List[DiscoveredRelationship],
        max_depth: Optional[int] = None,
    ) -> DiscoveryResult:
        """Recursively discover sub-tier suppliers.

        Starting from a root supplier, follows known relationships
        to discover suppliers at deeper tiers. Stops at max_depth
        or when no new suppliers are found. Detects and avoids
        circular references.

        Args:
            root_supplier: The starting (Tier 1) supplier.
            known_relationships: Known relationships to traverse.
            max_depth: Maximum recursion depth. Defaults to
                self.max_recursion_depth.

        Returns:
            DiscoveryResult with the full supplier tree discovered.
        """
        start_time = time.monotonic()
        effective_depth = (
            max_depth
            if max_depth is not None
            else self.max_recursion_depth
        )

        logger.info(
            "Starting recursive discovery: root=%s (%s), "
            "max_depth=%d, known_relationships=%d",
            root_supplier.name,
            root_supplier.supplier_id,
            effective_depth,
            len(known_relationships),
        )

        all_suppliers: List[DiscoveredSupplier] = [root_supplier]
        all_relationships: List[DiscoveredRelationship] = []
        visited_ids: Set[str] = {root_supplier.supplier_id}
        errors: List[str] = []
        warnings: List[str] = []

        # Build adjacency map: buyer_id -> list of supplier_ids
        adjacency: Dict[str, List[DiscoveredRelationship]] = (
            defaultdict(list)
        )
        for rel in known_relationships:
            adjacency[rel.buyer_id].append(rel)

        # BFS traversal to avoid deep recursion stack issues
        current_tier_ids: List[str] = [root_supplier.supplier_id]
        current_depth = 0

        while current_tier_ids and current_depth < effective_depth:
            next_tier_ids: List[str] = []
            current_depth += 1

            for buyer_id in current_tier_ids:
                upstream_rels = adjacency.get(buyer_id, [])
                for rel in upstream_rels:
                    if rel.supplier_id in visited_ids:
                        warnings.append(
                            f"Circular reference detected: "
                            f"{rel.supplier_id} already visited"
                        )
                        continue

                    visited_ids.add(rel.supplier_id)
                    next_tier_ids.append(rel.supplier_id)

                    # Create a placeholder supplier for discovered IDs
                    tier_supplier = DiscoveredSupplier(
                        supplier_id=rel.supplier_id,
                        name=rel.metadata.get("supplier_name", "")
                        if rel.metadata
                        else "",
                        commodity_types=[rel.commodity]
                        if rel.commodity
                        else [],
                        discovery_source=(
                            DiscoverySource.CROSS_REFERENCE.value
                        ),
                        metadata={"tier_level": current_depth},
                    )
                    all_suppliers.append(tier_supplier)

                    # Update relationship with discovered tier level
                    updated_rel = DiscoveredRelationship(
                        relationship_id=rel.relationship_id,
                        supplier_id=rel.supplier_id,
                        buyer_id=rel.buyer_id,
                        tier_level=current_depth,
                        commodity=rel.commodity,
                        confidence_score=rel.confidence_score,
                        confidence_level=rel.confidence_level,
                        discovery_source=rel.discovery_source,
                        volume_estimate_tonnes=(
                            rel.volume_estimate_tonnes
                        ),
                        evidence=rel.evidence,
                    )
                    all_relationships.append(updated_rel)

            current_tier_ids = next_tier_ids

        if current_depth >= effective_depth and current_tier_ids:
            warnings.append(
                f"Max depth {effective_depth} reached with "
                f"{len(current_tier_ids)} unexplored suppliers"
            )

        result = self._build_result(
            source=DiscoverySource.CROSS_REFERENCE.value,
            suppliers=all_suppliers,
            relationships=all_relationships,
            errors=errors,
            warnings=warnings,
            start_time=start_time,
        )

        logger.info(
            "Recursive discovery completed: root=%s, "
            "depth_reached=%d, suppliers=%d, "
            "relationships=%d, duration_ms=%.2f",
            root_supplier.name,
            current_depth,
            result.total_suppliers_found,
            result.total_relationships_found,
            result.processing_time_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate_suppliers(
        self, suppliers: List[DiscoveredSupplier]
    ) -> Tuple[List[DiscoveredSupplier], int]:
        """Fuzzy matching to identify and merge duplicate supplier entities.

        Uses normalized name similarity, country code matching, and
        registration ID exact matching to detect duplicates. When
        duplicates are found, the supplier with the most complete
        data is kept as the canonical entry.

        Args:
            suppliers: List of discovered suppliers to deduplicate.

        Returns:
            Tuple of (deduplicated supplier list, count of duplicates
            removed).
        """
        start_time = time.monotonic()
        logger.info(
            "Starting deduplication: input_count=%d, "
            "threshold=%.2f",
            len(suppliers),
            self.fuzzy_threshold,
        )

        if len(suppliers) <= 1:
            return suppliers, 0

        # Index by normalized name for efficient comparison
        normalized_names: List[str] = [
            _normalize_name(s.name) for s in suppliers
        ]

        # Track which indices are duplicates (map to canonical index)
        canonical_map: Dict[int, int] = {}
        duplicate_count = 0

        for i in range(len(suppliers)):
            if i in canonical_map:
                continue
            for j in range(i + 1, len(suppliers)):
                if j in canonical_map:
                    continue

                is_duplicate = self._check_duplicate(
                    suppliers[i],
                    suppliers[j],
                    normalized_names[i],
                    normalized_names[j],
                )
                if is_duplicate:
                    # Keep the one with more complete data
                    completeness_i = self._supplier_completeness(
                        suppliers[i]
                    )
                    completeness_j = self._supplier_completeness(
                        suppliers[j]
                    )

                    if completeness_j > completeness_i:
                        canonical_map[i] = j
                        duplicate_count += 1
                        break
                    else:
                        canonical_map[j] = i
                        duplicate_count += 1

        # Build deduplicated list
        deduplicated = [
            s
            for idx, s in enumerate(suppliers)
            if idx not in canonical_map
        ]

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Deduplication completed: input=%d, output=%d, "
            "duplicates=%d, duration_ms=%.2f",
            len(suppliers),
            len(deduplicated),
            duplicate_count,
            elapsed_ms,
        )
        return deduplicated, duplicate_count

    def _check_duplicate(
        self,
        supplier_a: DiscoveredSupplier,
        supplier_b: DiscoveredSupplier,
        norm_name_a: str,
        norm_name_b: str,
    ) -> bool:
        """Check if two suppliers are duplicates.

        Uses three criteria:
        1. Exact registration/tax/DUNS ID match (definitive)
        2. Fuzzy name match + same country (strong)
        3. Exact name match (moderate)

        Args:
            supplier_a: First supplier.
            supplier_b: Second supplier.
            norm_name_a: Normalized name of first supplier.
            norm_name_b: Normalized name of second supplier.

        Returns:
            True if suppliers are considered duplicates.
        """
        # Criterion 1: Exact ID match
        if (
            supplier_a.registration_id
            and supplier_a.registration_id == supplier_b.registration_id
        ):
            return True
        if (
            supplier_a.tax_id
            and supplier_a.tax_id == supplier_b.tax_id
        ):
            return True
        if (
            supplier_a.duns_number
            and supplier_a.duns_number == supplier_b.duns_number
        ):
            return True

        # Criterion 2: Fuzzy name + same country
        if norm_name_a and norm_name_b:
            similarity = _calculate_name_similarity(
                norm_name_a, norm_name_b
            )
            if similarity >= self.fuzzy_threshold:
                if (
                    supplier_a.country_code
                    and supplier_b.country_code
                    and supplier_a.country_code
                    == supplier_b.country_code
                ):
                    return True

        # Criterion 3: Exact normalized name
        if norm_name_a and norm_name_a == norm_name_b:
            return True

        return False

    def _supplier_completeness(
        self, supplier: DiscoveredSupplier
    ) -> int:
        """Calculate a simple completeness score for dedup priority.

        Args:
            supplier: Supplier to score.

        Returns:
            Integer completeness score (higher is more complete).
        """
        score = 0
        if supplier.name:
            score += 2
        if supplier.country_code:
            score += 2
        if supplier.registration_id:
            score += 3
        if supplier.tax_id:
            score += 3
        if supplier.duns_number:
            score += 2
        if supplier.gps_latitude is not None:
            score += 2
        if supplier.gps_longitude is not None:
            score += 2
        if supplier.address:
            score += 1
        if supplier.region:
            score += 1
        if supplier.commodity_types:
            score += 1
        return score

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def score_confidence(
        self, relationship: DiscoveredRelationship
    ) -> Tuple[float, str]:
        """Score discovery confidence for a supplier relationship.

        Calculates confidence score based on:
        1. Baseline confidence from discovery source
        2. Evidence count bonus
        3. Data completeness bonus
        4. Cross-source verification bonus

        The score is deterministic and reproducible.

        Args:
            relationship: The discovered relationship to score.

        Returns:
            Tuple of (confidence_score, confidence_level).
        """
        # Step 1: Baseline from source
        baseline = _SOURCE_BASELINE_CONFIDENCE.get(
            relationship.discovery_source,
            _SOURCE_BASELINE_CONFIDENCE[DiscoverySource.MANUAL.value],
        )

        # Step 2: Evidence count bonus (up to +0.10)
        evidence_count = len(relationship.evidence)
        evidence_bonus = min(0.10, evidence_count * 0.025)

        # Step 3: Volume data bonus
        volume_bonus = (
            0.05 if relationship.volume_estimate_tonnes is not None
            else 0.0
        )

        # Step 4: Metadata completeness bonus
        metadata_bonus = 0.0
        if relationship.metadata:
            metadata_fields = len(relationship.metadata)
            metadata_bonus = min(0.05, metadata_fields * 0.01)

        # Calculate final score, clamped to [0.0, 1.0]
        raw_score = (
            baseline + evidence_bonus + volume_bonus + metadata_bonus
        )
        final_score = max(0.0, min(1.0, raw_score))

        # Classify level
        confidence_level = _classify_confidence(final_score)

        logger.debug(
            "Confidence scored: relationship=%s, "
            "baseline=%.2f, evidence_bonus=%.2f, "
            "volume_bonus=%.2f, metadata_bonus=%.2f, "
            "final=%.2f, level=%s",
            relationship.relationship_id,
            baseline,
            evidence_bonus,
            volume_bonus,
            metadata_bonus,
            final_score,
            confidence_level,
        )

        return final_score, confidence_level

    # ------------------------------------------------------------------
    # Batch discovery
    # ------------------------------------------------------------------

    def batch_discover(
        self,
        suppliers: List[Dict[str, Any]],
        source: str = DiscoverySource.MANUAL.value,
    ) -> BatchDiscoveryResult:
        """Batch discovery across multiple root suppliers.

        Processes multiple supplier root nodes, running discovery
        from each one and aggregating results. Deduplicates across
        the entire batch.

        Args:
            suppliers: List of supplier data dictionaries, each
                representing a root node for discovery.
            source: Discovery source type for the batch.

        Returns:
            BatchDiscoveryResult with aggregated results.
        """
        start_time = time.monotonic()
        batch_id = str(uuid.uuid4())

        logger.info(
            "Starting batch discovery: batch_id=%s, "
            "input_count=%d, source=%s",
            batch_id,
            len(suppliers),
            source,
        )

        individual_results: List[DiscoveryResult] = []
        all_suppliers: List[DiscoveredSupplier] = []
        total_relationships = 0
        batch_errors: List[str] = []

        for idx, supplier_data in enumerate(suppliers):
            try:
                # Convert dict to DeclarationData-like structure
                declaration = DeclarationData(
                    declaration_id=f"batch-{batch_id}-{idx}",
                    declaring_supplier_id=str(
                        supplier_data.get("supplier_id", f"root-{idx}")
                    ),
                    declaring_supplier_name=str(
                        supplier_data.get("name", "")
                    ),
                    commodity=str(
                        supplier_data.get("commodity", "")
                    ),
                    declared_suppliers=supplier_data.get(
                        "sub_suppliers", []
                    ),
                    declaration_date=_utcnow_iso(),
                    declaration_type="batch",
                )

                result = self.discover_from_declaration(declaration)
                individual_results.append(result)
                all_suppliers.extend(result.discovered_suppliers)
                total_relationships += (
                    result.total_relationships_found
                )

                if result.errors:
                    batch_errors.extend(result.errors)

            except Exception as exc:
                error_msg = (
                    f"Batch item {idx} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                batch_errors.append(error_msg)
                logger.warning(error_msg)

        # Deduplicate across the entire batch
        deduplicated, dup_count = self.deduplicate_suppliers(
            all_suppliers
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        status = (
            DiscoveryStatus.COMPLETED.value
            if not batch_errors
            else DiscoveryStatus.PARTIAL.value
        )

        # Compute batch provenance
        provenance_data = {
            "batch_id": batch_id,
            "input_count": len(suppliers),
            "total_discovered": len(deduplicated),
            "total_relationships": total_relationships,
            "total_duplicates": dup_count,
            "source": source,
            "timestamp": _utcnow_iso(),
        }
        provenance_hash = _compute_provenance_hash(provenance_data)

        batch_result = BatchDiscoveryResult(
            batch_id=batch_id,
            status=status,
            total_input_suppliers=len(suppliers),
            total_discovered=len(deduplicated),
            total_relationships=total_relationships,
            total_duplicates=dup_count,
            individual_results=individual_results,
            processing_time_ms=elapsed_ms,
            provenance_hash=provenance_hash,
            errors=batch_errors,
        )

        logger.info(
            "Batch discovery completed: batch_id=%s, "
            "status=%s, discovered=%d, relationships=%d, "
            "duplicates=%d, errors=%d, duration_ms=%.2f",
            batch_id,
            status,
            len(deduplicated),
            total_relationships,
            dup_count,
            len(batch_errors),
            elapsed_ms,
        )
        return batch_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_relationship(
        self,
        supplier_id: str,
        buyer_id: str,
        commodity: str,
        source: str,
        tier_level: int = 2,
        volume: Optional[float] = None,
        evidence: Optional[List[str]] = None,
    ) -> DiscoveredRelationship:
        """Create a DiscoveredRelationship with confidence scoring.

        Args:
            supplier_id: Upstream supplier ID.
            buyer_id: Downstream buyer ID.
            commodity: Commodity traded.
            source: Discovery source type.
            tier_level: Tier level of the upstream supplier.
            volume: Volume estimate in tonnes.
            evidence: Evidence references.

        Returns:
            Scored DiscoveredRelationship instance.
        """
        relationship = DiscoveredRelationship(
            supplier_id=supplier_id,
            buyer_id=buyer_id,
            tier_level=tier_level,
            commodity=commodity,
            discovery_source=source,
            volume_estimate_tonnes=volume,
            evidence=evidence or [],
        )

        # Score confidence
        score, level = self.score_confidence(relationship)
        relationship.confidence_score = score
        relationship.confidence_level = level

        return relationship

    def _build_result(
        self,
        source: str,
        suppliers: List[DiscoveredSupplier],
        relationships: List[DiscoveredRelationship],
        errors: List[str],
        warnings: List[str],
        start_time: float,
    ) -> DiscoveryResult:
        """Build a DiscoveryResult with provenance hash.

        Args:
            source: Discovery source type.
            suppliers: Discovered supplier entities.
            relationships: Discovered relationships.
            errors: Errors encountered.
            warnings: Warnings generated.
            start_time: monotonic start time for duration calculation.

        Returns:
            DiscoveryResult with provenance hash and timing.
        """
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Determine status based on errors
        if errors and not suppliers:
            status = DiscoveryStatus.FAILED.value
        elif errors:
            status = DiscoveryStatus.PARTIAL.value
        else:
            status = DiscoveryStatus.COMPLETED.value

        # Compute provenance hash
        provenance_data = {
            "source": source,
            "supplier_count": len(suppliers),
            "relationship_count": len(relationships),
            "supplier_ids": sorted(s.supplier_id for s in suppliers),
            "timestamp": _utcnow_iso(),
        }
        provenance_hash = _compute_provenance_hash(provenance_data)

        return DiscoveryResult(
            status=status,
            discovered_suppliers=suppliers,
            discovered_relationships=relationships,
            discovery_source=source,
            total_suppliers_found=len(suppliers),
            total_relationships_found=len(relationships),
            duplicates_detected=0,
            processing_time_ms=elapsed_ms,
            provenance_hash=provenance_hash,
            errors=errors,
            warnings=warnings,
        )

    def _extract_commodity_list(
        self, data: Dict[str, Any], default_commodity: str
    ) -> List[str]:
        """Extract commodity list from a data dictionary.

        Args:
            data: Source dictionary that may contain commodity fields.
            default_commodity: Default commodity if none found.

        Returns:
            List of commodity strings.
        """
        commodities = data.get(
            "commodity_types",
            data.get("commodities", data.get("commodity")),
        )
        if isinstance(commodities, list):
            return [str(c).strip() for c in commodities if c]
        if isinstance(commodities, str) and commodities.strip():
            return [commodities.strip()]
        if default_commodity:
            return [default_commodity]
        return []

    def _extract_tier_level(self, data: Dict[str, Any]) -> int:
        """Extract tier level from a data dictionary.

        Args:
            data: Source dictionary.

        Returns:
            Tier level integer, defaults to 2.
        """
        tier = data.get("tier", data.get("tier_level"))
        if tier is not None:
            try:
                return max(1, int(tier))
            except (ValueError, TypeError):
                pass
        return 2

    def _extract_volume(
        self, data: Dict[str, Any]
    ) -> Optional[float]:
        """Extract volume from a data dictionary.

        Args:
            data: Source dictionary.

        Returns:
            Volume as float if present, None otherwise.
        """
        volume = data.get(
            "volume",
            data.get("volume_tonnes", data.get("quantity")),
        )
        return self._safe_float(volume)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert a value to float.

        Args:
            value: Value to convert.

        Returns:
            Float value or None if conversion fails.
        """
        if value is None:
            return None
        try:
            result = float(value)
            # Reject NaN and infinity
            if result != result or result == float("inf"):
                return None
            return result
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Metrics accessors
    # ------------------------------------------------------------------

    @property
    def total_discoveries(self) -> int:
        """Return total number of suppliers discovered.

        Returns:
            Running count of discovered suppliers.
        """
        return self._discovery_count

    @property
    def total_relationships(self) -> int:
        """Return total number of relationships discovered.

        Returns:
            Running count of discovered relationships.
        """
        return self._relationship_count

    def reset_metrics(self) -> None:
        """Reset internal metrics counters to zero."""
        self._discovery_count = 0
        self._relationship_count = 0
        logger.debug("SupplierDiscoveryEngine metrics reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Engine
    "SupplierDiscoveryEngine",
    # Enums
    "DiscoverySource",
    "ConfidenceLevel",
    "DiscoveryStatus",
    "EUDRCommodity",
    # Data classes
    "DiscoveredSupplier",
    "DiscoveredRelationship",
    "DeclarationData",
    "QuestionnaireData",
    "ShippingData",
    "CertificationData",
    "ERPData",
    "DiscoveryResult",
    "BatchDiscoveryResult",
    # Constants
    "MAX_RECURSION_DEPTH",
    "DEFAULT_BATCH_SIZE",
    "FUZZY_MATCH_THRESHOLD",
    "MIN_CONFIDENCE_THRESHOLD",
    "ENGINE_VERSION",
    "METRIC_PREFIX",
]
