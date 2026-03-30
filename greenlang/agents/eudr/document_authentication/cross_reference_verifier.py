# -*- coding: utf-8 -*-
"""
Cross-Reference Verifier - AGENT-EUDR-012 Engine 7

Verify documents against external certification registries and
cross-reference document data against supply chain records, mass
balance ledgers, and shipping documents. Includes response caching
with configurable TTL and per-registry rate limiting.

Zero-Hallucination Guarantees:
    - All verification uses deterministic string/date/quantity matching
    - No ML/LLM used for registry verification decisions
    - Cache hits are served deterministically from in-memory store
    - Rate limiting uses simple counter-based sliding window
    - SHA-256 provenance hashes on every verification operation
    - All discrepancy detection uses configurable thresholds

Registry Integrations (simulated in v1.0, production-ready interface):
    - FSC: Forest Stewardship Council certificate database (info.fsc.org)
    - RSPO: Roundtable on Sustainable Palm Oil (PalmTrace)
    - ISCC: International Sustainability & Carbon Certification
    - Fairtrade: Fairtrade International / FLOCERT database
    - UTZ/RA: UTZ / Rainforest Alliance certification portal
    - IPPC: International Plant Protection Convention (ePhyto system)

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Due diligence obligations
    - EU 2023/1115 (EUDR) Article 10: Risk assessment and mitigation
    - EU 2023/1115 (EUDR) Article 14: Five-year record retention
    - EU 2023/1115 (EUDR) Article 21: Competent authority verification
    - ISO 22095:2020: Chain of Custody requirements

Performance Targets:
    - Cache hit: <5ms
    - Single registry verification (simulated): <50ms
    - Batch of 50 verifications: <2s
    - Cache statistics: <1ms

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.document_authentication.config import (
    DocumentAuthenticationConfig,
    get_config,
)
from greenlang.agents.eudr.document_authentication.models import (
    CrossRefResult,
    RegistryType,
)
from greenlang.agents.eudr.document_authentication.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.schemas import utcnow
from greenlang.agents.eudr.document_authentication.metrics import (
    observe_crossref_duration,
    record_api_error,
    record_crossref_query,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str = "XREF") -> str:
    """Generate a prefixed UUID4 string identifier.

    Args:
        prefix: String prefix for the identifier.

    Returns:
        Prefixed UUID4 string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

# ---------------------------------------------------------------------------
# Simulated registry data for v1.0
# ---------------------------------------------------------------------------

#: Simulated FSC certificate records (for deterministic testing).
_SIMULATED_FSC_RECORDS: Dict[str, Dict[str, Any]] = {
    "FSC-C000001": {
        "holder_name": "GreenTimber Corp",
        "status": "valid",
        "valid_from": "2023-01-01",
        "valid_to": "2028-12-31",
        "scope": "FSC Chain of Custody",
        "country": "BR",
        "commodities": ["wood"],
    },
    "FSC-C123456": {
        "holder_name": "Amazon Wood Products",
        "status": "valid",
        "valid_from": "2024-06-15",
        "valid_to": "2029-06-14",
        "scope": "FSC Chain of Custody",
        "country": "BR",
        "commodities": ["wood"],
    },
    "FSC-C999999": {
        "holder_name": "ExpiredTimber Inc",
        "status": "expired",
        "valid_from": "2018-01-01",
        "valid_to": "2023-01-01",
        "scope": "FSC Chain of Custody",
        "country": "ID",
        "commodities": ["wood"],
    },
}

#: Simulated RSPO certificate records.
_SIMULATED_RSPO_RECORDS: Dict[str, Dict[str, Any]] = {
    "RSPO-0000001": {
        "holder_name": "SustainPalm Sdn Bhd",
        "status": "valid",
        "valid_from": "2023-03-01",
        "valid_to": "2028-02-28",
        "scope": "RSPO P&C",
        "country": "MY",
        "commodities": ["oil_palm"],
    },
    "RSPO-1234567": {
        "holder_name": "PalmOil Global Ltd",
        "status": "valid",
        "valid_from": "2024-01-01",
        "valid_to": "2029-12-31",
        "scope": "RSPO SCCS",
        "country": "ID",
        "commodities": ["oil_palm"],
    },
}

#: Simulated ISCC certificate records.
_SIMULATED_ISCC_RECORDS: Dict[str, Dict[str, Any]] = {
    "ISCC-CERT-DE001-20240101": {
        "holder_name": "BioFuels GmbH",
        "status": "valid",
        "valid_from": "2024-01-01",
        "valid_to": "2027-12-31",
        "scope": "ISCC EU Certification",
        "country": "DE",
        "commodities": ["oil_palm", "soya"],
    },
}

#: Simulated Fairtrade records.
_SIMULATED_FAIRTRADE_RECORDS: Dict[str, Dict[str, Any]] = {
    "FLO-ID-12345": {
        "holder_name": "CocoaFair Cooperative",
        "status": "valid",
        "valid_from": "2023-06-01",
        "valid_to": "2026-05-31",
        "scope": "Fairtrade Cocoa",
        "country": "GH",
        "commodities": ["cocoa"],
    },
}

#: Simulated UTZ/Rainforest Alliance records.
_SIMULATED_UTZ_RA_RECORDS: Dict[str, Dict[str, Any]] = {
    "RA-CERT-98765": {
        "holder_name": "CoffeeEstate SA",
        "status": "valid",
        "valid_from": "2024-01-01",
        "valid_to": "2027-12-31",
        "scope": "Rainforest Alliance Coffee",
        "country": "CO",
        "commodities": ["coffee"],
    },
}

#: Simulated IPPC ePhyto records.
_SIMULATED_IPPC_RECORDS: Dict[str, Dict[str, Any]] = {
    "ePhyto-20240001000": {
        "holder_name": "Export Authority Brazil",
        "status": "valid",
        "valid_from": "2024-03-01",
        "valid_to": "2024-09-01",
        "scope": "Phytosanitary Certificate",
        "country": "BR",
        "commodities": ["coffee", "soya", "wood"],
    },
}

#: Registry type to simulated data mapping.
_REGISTRY_DATA: Dict[str, Dict[str, Dict[str, Any]]] = {
    "fsc": _SIMULATED_FSC_RECORDS,
    "rspo": _SIMULATED_RSPO_RECORDS,
    "iscc": _SIMULATED_ISCC_RECORDS,
    "fairtrade": _SIMULATED_FAIRTRADE_RECORDS,
    "utz_ra": _SIMULATED_UTZ_RA_RECORDS,
    "ippc": _SIMULATED_IPPC_RECORDS,
}

#: Lab accreditation bodies for verification.
_ACCREDITED_LABS: Dict[str, List[str]] = {
    "BR": ["INMETRO", "ABNT", "MAPA-LAB"],
    "ID": ["KAN", "BSN"],
    "MY": ["DSM", "SIRIM"],
    "DE": ["DAkkS", "BAM"],
    "DEFAULT": ["ISO 17025", "ILAC"],
}

# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

class _CacheEntry:
    """Internal cache entry for cross-reference results.

    Attributes:
        result: Cached verification result dictionary.
        created_at: UTC datetime when the entry was cached.
        expires_at: UTC datetime when the entry expires.
        hits: Number of cache hits for this entry.
    """

    __slots__ = ("result", "created_at", "expires_at", "hits")

    def __init__(
        self,
        result: Dict[str, Any],
        ttl_hours: int,
    ) -> None:
        now = utcnow()
        self.result = result
        self.created_at = now
        self.expires_at = now + timedelta(hours=ttl_hours)
        self.hits = 0

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return utcnow() > self.expires_at

# ---------------------------------------------------------------------------
# CrossReferenceVerifier
# ---------------------------------------------------------------------------

class CrossReferenceVerifier:
    """Cross-reference verification engine for EUDR document authentication.

    Verifies document certificates against external registries (FSC,
    RSPO, ISCC, Fairtrade, UTZ/RA, IPPC) and cross-references document
    data against supply chain records, mass balance ledgers, and
    shipping documents.

    Includes in-memory response caching with configurable TTL and
    per-registry rate limiting for production resilience.

    All operations are thread-safe via reentrant locking. All verification
    uses deterministic matching for zero-hallucination compliance.

    Attributes:
        _config: Document authentication configuration.
        _provenance: ProvenanceTracker for audit trail.
        _cache: In-memory cache keyed by cache key string.
        _rate_counters: Per-registry rate limit counters.
        _results: In-memory verification result storage.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> verifier = CrossReferenceVerifier()
        >>> result = verifier.verify_against_registry(
        ...     document_id="doc-001",
        ...     certificate_number="FSC-C123456",
        ...     registry_type="fsc",
        ... )
        >>> assert result["success"] is True
    """

    def __init__(
        self,
        config: Optional[DocumentAuthenticationConfig] = None,
    ) -> None:
        """Initialize CrossReferenceVerifier.

        Args:
            config: Optional configuration override. If None, the
                singleton configuration from ``get_config()`` is used.
        """
        self._config: DocumentAuthenticationConfig = config or get_config()
        self._provenance: ProvenanceTracker = get_provenance_tracker()

        # In-memory cache
        self._cache: Dict[str, _CacheEntry] = {}

        # Rate limiting counters: registry -> {minute_key: count}
        self._rate_counters: Dict[str, Dict[str, int]] = {}

        # Result storage
        self._results: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "CrossReferenceVerifier initialized: "
            "cache_ttl=%dh, fsc_rate=%d/min, rspo_rate=%d/min, "
            "iscc_rate=%d/min, timeout=%ds",
            self._config.cache_ttl_hours,
            self._config.fsc_api_rate_limit,
            self._config.rspo_api_rate_limit,
            self._config.iscc_api_rate_limit,
            self._config.crossref_timeout_s,
        )

    # ------------------------------------------------------------------
    # Public API: Verify against registry
    # ------------------------------------------------------------------

    def verify_against_registry(
        self,
        document_id: str,
        certificate_number: str,
        registry_type: str,
        holder_name: Optional[str] = None,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
        commodity: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Verify a certificate against an external registry.

        Performs a lookup of the certificate number in the specified
        registry and cross-references the returned data against the
        claimed attributes (holder name, validity dates, scope).

        Args:
            document_id: Document identifier for provenance tracking.
            certificate_number: Certificate/reference number to look up.
            registry_type: Registry to query (fsc, rspo, iscc, fairtrade,
                utz_ra, ippc).
            holder_name: Expected certificate holder name for matching.
            valid_from: Expected validity start date.
            valid_to: Expected validity end date.
            commodity: EUDR commodity for scope matching.
            use_cache: Whether to use cached results.

        Returns:
            Dictionary with keys: success, crossref_id, document_id,
            registry_type, certificate_number, registry_found,
            registry_status, registry_data, name_match, date_match,
            scope_match, discrepancies, cached, processing_time_ms,
            provenance_hash.

        Raises:
            ValueError: If document_id or certificate_number is empty.
        """
        start_time = time.monotonic()

        if not document_id:
            raise ValueError("document_id must not be empty")
        if not certificate_number:
            raise ValueError("certificate_number must not be empty")

        crossref_id = _generate_id("XREF")
        registry_lower = registry_type.lower().strip()

        logger.info(
            "Verifying against registry: document_id=%s, "
            "cert=%s, registry=%s",
            document_id[:16], certificate_number[:20], registry_lower,
        )

        try:
            # Step 1: Check cache
            if use_cache:
                cache_key = self._build_cache_key(
                    certificate_number, registry_lower,
                )
                cached = self._get_cached_result(cache_key)
                if cached is not None:
                    elapsed_ms = (time.monotonic() - start_time) * 1000
                    cached["cached"] = True
                    cached["crossref_id"] = crossref_id
                    cached["document_id"] = document_id
                    cached["processing_time_ms"] = round(elapsed_ms, 2)

                    logger.info(
                        "Cache hit: document_id=%s, cert=%s, "
                        "registry=%s, elapsed=%.1fms",
                        document_id[:16], certificate_number[:20],
                        registry_lower, elapsed_ms,
                    )
                    return cached

            # Step 2: Check rate limit
            if not self._check_rate_limit(registry_lower):
                elapsed_ms = (time.monotonic() - start_time) * 1000
                return self._build_rate_limited_result(
                    crossref_id, document_id, certificate_number,
                    registry_lower, elapsed_ms,
                )

            # Step 3: Query registry
            registry_data = self._query_registry(
                certificate_number, registry_lower,
            )

            # Step 4: Determine if found
            registry_found = registry_data is not None

            # Step 5: Cross-reference attributes
            discrepancies: List[str] = []
            name_match: Optional[bool] = None
            date_match: Optional[bool] = None
            scope_match: Optional[bool] = None
            registry_status: Optional[str] = None
            registry_holder: Optional[str] = None
            registry_valid_from: Optional[str] = None
            registry_valid_to: Optional[str] = None
            registry_scope: Optional[str] = None

            if registry_found and registry_data:
                registry_status = registry_data.get("status")
                registry_holder = registry_data.get("holder_name")
                registry_valid_from = registry_data.get("valid_from")
                registry_valid_to = registry_data.get("valid_to")
                registry_scope = registry_data.get("scope")

                # Name match
                if holder_name and registry_holder:
                    name_match = self._names_match(
                        holder_name, registry_holder,
                    )
                    if not name_match:
                        discrepancies.append(
                            f"Holder name mismatch: claimed "
                            f"'{holder_name}' vs registry "
                            f"'{registry_holder}'"
                        )

                # Date match
                if valid_from and registry_valid_from:
                    reg_from = self._parse_date(registry_valid_from)
                    if reg_from and valid_from:
                        claimed_from = self._ensure_datetime(valid_from)
                        diff_days = abs((claimed_from - reg_from).days)
                        date_match = diff_days <= 7
                        if not date_match:
                            discrepancies.append(
                                f"Validity start mismatch: claimed "
                                f"{claimed_from.date()} vs registry "
                                f"{reg_from.date()} "
                                f"(diff={diff_days} days)"
                            )

                if valid_to and registry_valid_to:
                    reg_to = self._parse_date(registry_valid_to)
                    if reg_to and valid_to:
                        claimed_to = self._ensure_datetime(valid_to)
                        diff_days = abs((claimed_to - reg_to).days)
                        if diff_days > 7:
                            date_match = False
                            discrepancies.append(
                                f"Validity end mismatch: claimed "
                                f"{claimed_to.date()} vs registry "
                                f"{reg_to.date()} "
                                f"(diff={diff_days} days)"
                            )
                        elif date_match is None:
                            date_match = True

                # Scope / commodity match
                if commodity and registry_data.get("commodities"):
                    reg_commodities = registry_data["commodities"]
                    scope_match = commodity.lower() in [
                        c.lower() for c in reg_commodities
                    ]
                    if not scope_match:
                        discrepancies.append(
                            f"Commodity scope mismatch: "
                            f"'{commodity}' not in registry scope "
                            f"{reg_commodities}"
                        )

                # Check certificate status
                if registry_status == "expired":
                    discrepancies.append(
                        "Certificate is expired according to the registry"
                    )
                elif registry_status == "revoked":
                    discrepancies.append(
                        "Certificate has been revoked by the registry"
                    )
                elif registry_status == "suspended":
                    discrepancies.append(
                        "Certificate is currently suspended"
                    )

            elif not registry_found:
                discrepancies.append(
                    f"Certificate '{certificate_number}' not found in "
                    f"{registry_lower.upper()} registry"
                )

            # Step 6: Build result
            elapsed_ms = (time.monotonic() - start_time) * 1000

            result: Dict[str, Any] = {
                "success": True,
                "crossref_id": crossref_id,
                "document_id": document_id,
                "registry_type": registry_lower,
                "certificate_number": certificate_number,
                "registry_found": registry_found,
                "registry_status": registry_status,
                "registry_holder_name": registry_holder,
                "registry_valid_from": registry_valid_from,
                "registry_valid_to": registry_valid_to,
                "registry_scope": registry_scope,
                "registry_data": registry_data,
                "name_match": name_match,
                "date_match": date_match,
                "scope_match": scope_match,
                "discrepancies": discrepancies,
                "cached": False,
                "processing_time_ms": round(elapsed_ms, 2),
            }

            # Step 7: Compute provenance
            provenance_data = {
                "crossref_id": crossref_id,
                "document_id": document_id,
                "registry_type": registry_lower,
                "certificate_number": certificate_number,
                "registry_found": registry_found,
                "discrepancy_count": len(discrepancies),
                "module_version": _MODULE_VERSION,
            }
            provenance_hash = _compute_hash(provenance_data)
            result["provenance_hash"] = provenance_hash

            if self._config.enable_provenance:
                self._provenance.record(
                    entity_type="crossref",
                    action="cross_reference",
                    entity_id=document_id,
                    data=provenance_data,
                    metadata={
                        "crossref_id": crossref_id,
                        "document_id": document_id,
                        "registry_type": registry_lower,
                        "registry_found": registry_found,
                    },
                )

            # Step 8: Cache result
            if use_cache:
                self._cache_result(cache_key, result)

            # Step 9: Store and record metrics
            with self._lock:
                self._results[crossref_id] = result

            if self._config.enable_metrics:
                observe_crossref_duration(elapsed_ms / 1000)
                record_crossref_query(registry_lower)

            logger.info(
                "Registry verification completed: document_id=%s, "
                "cert=%s, registry=%s, found=%s, discrepancies=%d, "
                "elapsed=%.1fms",
                document_id[:16], certificate_number[:20],
                registry_lower, registry_found,
                len(discrepancies), elapsed_ms,
            )

            return result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Registry verification failed: document_id=%s, "
                "cert=%s, registry=%s, error=%s",
                document_id[:16], certificate_number[:20],
                registry_lower, str(exc), exc_info=True,
            )
            if self._config.enable_metrics:
                record_api_error("cross_reference")
            return {
                "success": False,
                "crossref_id": crossref_id,
                "document_id": document_id,
                "registry_type": registry_lower,
                "certificate_number": certificate_number,
                "registry_found": False,
                "registry_status": None,
                "discrepancies": [f"Verification error: {str(exc)}"],
                "cached": False,
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": None,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Public API: Batch verify
    # ------------------------------------------------------------------

    def batch_verify(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Verify multiple certificates against external registries.

        Args:
            documents: List of dictionaries, each containing at minimum:
                document_id (str), certificate_number (str),
                registry_type (str). Optional: holder_name, valid_from,
                valid_to, commodity, use_cache.

        Returns:
            List of verification result dictionaries.

        Raises:
            ValueError: If documents list is empty or exceeds batch limit.
        """
        if not documents:
            raise ValueError("documents list must not be empty")

        max_size = self._config.batch_max_size
        if len(documents) > max_size:
            raise ValueError(
                f"Batch size {len(documents)} exceeds maximum {max_size}"
            )

        logger.info(
            "Batch cross-reference verification: %d documents",
            len(documents),
        )

        results: List[Dict[str, Any]] = []
        for doc in documents:
            result = self.verify_against_registry(
                document_id=doc.get("document_id", str(uuid.uuid4())),
                certificate_number=doc.get("certificate_number", ""),
                registry_type=doc.get("registry_type", "fsc"),
                holder_name=doc.get("holder_name"),
                valid_from=doc.get("valid_from"),
                valid_to=doc.get("valid_to"),
                commodity=doc.get("commodity"),
                use_cache=doc.get("use_cache", True),
            )
            results.append(result)

        found_count = sum(
            1 for r in results if r.get("registry_found")
        )
        logger.info(
            "Batch verification completed: %d/%d found",
            found_count, len(documents),
        )
        return results

    # ------------------------------------------------------------------
    # Public API: Cross-reference parties
    # ------------------------------------------------------------------

    def cross_reference_parties(
        self,
        document_id: str,
        document_parties: Dict[str, str],
        supply_chain_parties: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Cross-reference document parties against supply chain graph.

        Args:
            document_id: Document identifier.
            document_parties: Parties named in the document (e.g.,
                exporter, importer, consignee).
            supply_chain_parties: Known parties in the supply chain.

        Returns:
            Dictionary with matched_parties, unmatched_parties,
            match_rate, discrepancies.
        """
        start_time = time.monotonic()

        matched: List[str] = []
        unmatched: List[str] = []
        discrepancies: List[str] = []

        sc_names = set()
        for party in supply_chain_parties:
            name = party.get("name", "")
            if name:
                sc_names.add(name.lower().strip())

        for role, name in document_parties.items():
            if not name:
                continue
            name_lower = name.lower().strip()
            found = any(
                name_lower in sc_name or sc_name in name_lower
                for sc_name in sc_names
            )
            if found:
                matched.append(role)
            else:
                unmatched.append(role)
                discrepancies.append(
                    f"Party '{name}' (role: {role}) not found in "
                    f"supply chain graph"
                )

        total = len(matched) + len(unmatched)
        match_rate = (
            len(matched) / total * 100.0 if total > 0 else 0.0
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "document_id": document_id,
            "matched_parties": matched,
            "unmatched_parties": unmatched,
            "match_rate": round(match_rate, 1),
            "discrepancies": discrepancies,
            "processing_time_ms": round(elapsed_ms, 2),
            "provenance_hash": _compute_hash({
                "document_id": document_id,
                "matched": matched,
                "unmatched": unmatched,
            }),
        }

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="crossref",
                action="cross_reference",
                entity_id=document_id,
                data=result,
                metadata={
                    "document_id": document_id,
                    "match_rate": match_rate,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Public API: Cross-reference quantities
    # ------------------------------------------------------------------

    def cross_reference_quantities(
        self,
        document_id: str,
        document_quantity: float,
        ledger_quantity: float,
        unit: str = "kg",
    ) -> Dict[str, Any]:
        """Cross-reference document quantity against mass balance ledger.

        Args:
            document_id: Document identifier.
            document_quantity: Quantity stated in the document.
            ledger_quantity: Quantity recorded in the mass balance ledger.
            unit: Unit of measurement.

        Returns:
            Dictionary with match, variance_percent, variance_absolute,
            within_tolerance, discrepancy.
        """
        start_time = time.monotonic()

        tolerance_pct = self._config.quantity_tolerance_percent

        if ledger_quantity == 0:
            variance_pct = 100.0 if document_quantity != 0 else 0.0
        else:
            variance_pct = abs(
                (document_quantity - ledger_quantity) / ledger_quantity
            ) * 100.0

        variance_abs = abs(document_quantity - ledger_quantity)
        within_tolerance = variance_pct <= tolerance_pct

        discrepancy = None
        if not within_tolerance:
            discrepancy = (
                f"Quantity mismatch: document={document_quantity:.2f} "
                f"{unit}, ledger={ledger_quantity:.2f} {unit}, "
                f"variance={variance_pct:.1f}% "
                f"(tolerance={tolerance_pct:.1f}%)"
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "document_id": document_id,
            "document_quantity": document_quantity,
            "ledger_quantity": ledger_quantity,
            "unit": unit,
            "match": within_tolerance,
            "variance_percent": round(variance_pct, 2),
            "variance_absolute": round(variance_abs, 4),
            "within_tolerance": within_tolerance,
            "tolerance_percent": tolerance_pct,
            "discrepancy": discrepancy,
            "processing_time_ms": round(elapsed_ms, 2),
            "provenance_hash": _compute_hash({
                "document_id": document_id,
                "doc_qty": document_quantity,
                "ledger_qty": ledger_quantity,
            }),
        }

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="crossref",
                action="cross_reference",
                entity_id=document_id,
                data=result,
                metadata={
                    "document_id": document_id,
                    "within_tolerance": within_tolerance,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Public API: Cross-reference dates
    # ------------------------------------------------------------------

    def cross_reference_dates(
        self,
        document_id: str,
        document_dates: Dict[str, Any],
        reference_dates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Cross-reference document dates against shipping/customs records.

        Args:
            document_id: Document identifier.
            document_dates: Dates from the document (e.g., issuance_date,
                shipping_date, arrival_date).
            reference_dates: Reference dates from BOL, customs declarations.

        Returns:
            Dictionary with date_comparisons, discrepancies, all_match.
        """
        start_time = time.monotonic()
        tolerance_days = self._config.date_tolerance_days

        comparisons: List[Dict[str, Any]] = []
        discrepancies: List[str] = []

        for key in document_dates:
            if key not in reference_dates:
                continue

            doc_date = self._parse_date(document_dates[key])
            ref_date = self._parse_date(reference_dates[key])

            if not doc_date or not ref_date:
                continue

            diff_days = abs((doc_date - ref_date).days)
            match = diff_days <= tolerance_days

            comparison = {
                "field": key,
                "document_date": doc_date.isoformat(),
                "reference_date": ref_date.isoformat(),
                "diff_days": diff_days,
                "match": match,
                "tolerance_days": tolerance_days,
            }
            comparisons.append(comparison)

            if not match:
                discrepancies.append(
                    f"Date mismatch for '{key}': document "
                    f"{doc_date.date()} vs reference {ref_date.date()} "
                    f"(diff={diff_days} days, tolerance={tolerance_days})"
                )

        all_match = len(discrepancies) == 0 and len(comparisons) > 0
        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "document_id": document_id,
            "date_comparisons": comparisons,
            "discrepancies": discrepancies,
            "all_match": all_match,
            "processing_time_ms": round(elapsed_ms, 2),
            "provenance_hash": _compute_hash({
                "document_id": document_id,
                "comparisons": comparisons,
            }),
        }

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="crossref",
                action="cross_reference",
                entity_id=document_id,
                data=result,
                metadata={
                    "document_id": document_id,
                    "all_match": all_match,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Public API: Verify lab accreditation
    # ------------------------------------------------------------------

    def verify_lab_accreditation(
        self,
        document_id: str,
        lab_name: str,
        country: str,
        accreditation_body: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify laboratory accreditation for test certificates.

        Args:
            document_id: Document identifier.
            lab_name: Name of the testing laboratory.
            country: Country where the lab is located.
            accreditation_body: Claimed accreditation body.

        Returns:
            Dictionary with accredited, accreditation_body_match,
            recognized_bodies, discrepancies.
        """
        start_time = time.monotonic()

        recognized = _ACCREDITED_LABS.get(
            country.upper(), _ACCREDITED_LABS.get("DEFAULT", []),
        )

        accredited = False
        body_match = None
        discrepancies: List[str] = []

        if accreditation_body:
            body_lower = accreditation_body.lower()
            body_match = any(
                r.lower() in body_lower or body_lower in r.lower()
                for r in recognized
            )
            accredited = body_match

            if not body_match:
                discrepancies.append(
                    f"Accreditation body '{accreditation_body}' is not "
                    f"recognized for country '{country}'. Recognized: "
                    f"{recognized}"
                )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "document_id": document_id,
            "lab_name": lab_name,
            "country": country,
            "accredited": accredited,
            "accreditation_body_claimed": accreditation_body,
            "accreditation_body_match": body_match,
            "recognized_bodies": recognized,
            "discrepancies": discrepancies,
            "processing_time_ms": round(elapsed_ms, 2),
            "provenance_hash": _compute_hash({
                "document_id": document_id,
                "lab_name": lab_name,
                "accredited": accredited,
            }),
        }

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="crossref",
                action="cross_reference",
                entity_id=document_id,
                data=result,
                metadata={
                    "document_id": document_id,
                    "accredited": accredited,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Public API: Cache statistics
    # ------------------------------------------------------------------

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dictionary with total_entries, expired_entries,
            total_hits, hit_rate, memory_entries_by_registry.
        """
        with self._lock:
            entries = list(self._cache.values())
            total = len(entries)

        expired = sum(1 for e in entries if e.is_expired())
        total_hits = sum(e.hits for e in entries)

        registry_counts: Dict[str, int] = {}
        for key in self._cache:
            parts = key.split(":", 1)
            if len(parts) >= 1:
                reg = parts[0]
                registry_counts[reg] = registry_counts.get(reg, 0) + 1

        return {
            "total_entries": total,
            "active_entries": total - expired,
            "expired_entries": expired,
            "total_hits": total_hits,
            "entries_by_registry": registry_counts,
            "cache_ttl_hours": self._config.cache_ttl_hours,
        }

    def clear_cache(self) -> int:
        """Clear all cached cross-reference results.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
        logger.info("Cross-reference cache cleared: %d entries removed", count)
        return count

    def evict_expired(self) -> int:
        """Remove expired entries from the cache.

        Returns:
            Number of expired entries removed.
        """
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
        if expired_keys:
            logger.info(
                "Cache eviction: %d expired entries removed",
                len(expired_keys),
            )
        return len(expired_keys)

    # ------------------------------------------------------------------
    # Internal: Registry-specific verification
    # ------------------------------------------------------------------

    def _verify_fsc(
        self,
        certificate_number: str,
    ) -> Optional[Dict[str, Any]]:
        """Verify a certificate against the FSC database.

        Args:
            certificate_number: FSC certificate/license number.

        Returns:
            Registry record or None if not found.
        """
        return _SIMULATED_FSC_RECORDS.get(certificate_number)

    def _verify_rspo(
        self,
        certificate_number: str,
    ) -> Optional[Dict[str, Any]]:
        """Verify a certificate against the RSPO PalmTrace database.

        Args:
            certificate_number: RSPO membership/certificate number.

        Returns:
            Registry record or None if not found.
        """
        return _SIMULATED_RSPO_RECORDS.get(certificate_number)

    def _verify_iscc(
        self,
        certificate_number: str,
    ) -> Optional[Dict[str, Any]]:
        """Verify a certificate against the ISCC certificate search.

        Args:
            certificate_number: ISCC certificate identifier.

        Returns:
            Registry record or None if not found.
        """
        return _SIMULATED_ISCC_RECORDS.get(certificate_number)

    def _verify_fairtrade(
        self,
        certificate_number: str,
    ) -> Optional[Dict[str, Any]]:
        """Verify a certificate against the Fairtrade/FLOCERT database.

        Args:
            certificate_number: Fairtrade/FLOCERT ID.

        Returns:
            Registry record or None if not found.
        """
        return _SIMULATED_FAIRTRADE_RECORDS.get(certificate_number)

    def _verify_utz_ra(
        self,
        certificate_number: str,
    ) -> Optional[Dict[str, Any]]:
        """Verify against the UTZ/Rainforest Alliance portal.

        Args:
            certificate_number: UTZ/RA certificate number.

        Returns:
            Registry record or None if not found.
        """
        return _SIMULATED_UTZ_RA_RECORDS.get(certificate_number)

    def _verify_ippc(
        self,
        certificate_number: str,
    ) -> Optional[Dict[str, Any]]:
        """Verify against the IPPC ePhyto system.

        Args:
            certificate_number: ePhyto number.

        Returns:
            Registry record or None if not found.
        """
        return _SIMULATED_IPPC_RECORDS.get(certificate_number)

    # ------------------------------------------------------------------
    # Internal: Query registry dispatcher
    # ------------------------------------------------------------------

    def _query_registry(
        self,
        certificate_number: str,
        registry_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Dispatch certificate lookup to the appropriate registry.

        Args:
            certificate_number: Certificate number to look up.
            registry_type: Registry identifier.

        Returns:
            Registry record dictionary or None.
        """
        dispatch = {
            "fsc": self._verify_fsc,
            "rspo": self._verify_rspo,
            "iscc": self._verify_iscc,
            "fairtrade": self._verify_fairtrade,
            "utz_ra": self._verify_utz_ra,
            "ippc": self._verify_ippc,
        }

        verify_fn = dispatch.get(registry_type)
        if not verify_fn:
            logger.warning(
                "Unknown registry type: %s", registry_type,
            )
            return None

        return verify_fn(certificate_number)

    # ------------------------------------------------------------------
    # Internal: Cache operations
    # ------------------------------------------------------------------

    def _build_cache_key(
        self,
        certificate_number: str,
        registry_type: str,
    ) -> str:
        """Build a deterministic cache key.

        Args:
            certificate_number: Certificate number.
            registry_type: Registry type.

        Returns:
            Cache key string.
        """
        return f"{registry_type}:{certificate_number}"

    def _get_cached_result(
        self,
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached cross-reference result if valid.

        Args:
            cache_key: Cache key string.

        Returns:
            Cached result dictionary or None if not found/expired.
        """
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._cache[cache_key]
                return None
            entry.hits += 1
            return dict(entry.result)

    def _cache_result(
        self,
        cache_key: str,
        result: Dict[str, Any],
    ) -> None:
        """Store a cross-reference result in the cache.

        Args:
            cache_key: Cache key string.
            result: Result dictionary to cache.
        """
        ttl = self._config.cache_ttl_hours
        with self._lock:
            self._cache[cache_key] = _CacheEntry(
                result=dict(result), ttl_hours=ttl,
            )

    # ------------------------------------------------------------------
    # Internal: Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(
        self,
        registry_type: str,
    ) -> bool:
        """Check if the current request is within the rate limit.

        Uses a per-minute sliding window counter.

        Args:
            registry_type: Registry type to check.

        Returns:
            True if within rate limit, False if exceeded.
        """
        limit = self._config.get_registry_rate_limit(registry_type)
        now = utcnow()
        minute_key = now.strftime("%Y-%m-%dT%H:%M")

        with self._lock:
            if registry_type not in self._rate_counters:
                self._rate_counters[registry_type] = {}
            counters = self._rate_counters[registry_type]

            # Clean old minute keys (keep only current minute)
            old_keys = [
                k for k in counters if k != minute_key
            ]
            for k in old_keys:
                del counters[k]

            current = counters.get(minute_key, 0)
            if current >= limit:
                logger.warning(
                    "Rate limit exceeded for registry %s: "
                    "%d/%d per minute",
                    registry_type, current, limit,
                )
                return False

            counters[minute_key] = current + 1
            return True

    # ------------------------------------------------------------------
    # Internal: Rate-limited result
    # ------------------------------------------------------------------

    def _build_rate_limited_result(
        self,
        crossref_id: str,
        document_id: str,
        certificate_number: str,
        registry_type: str,
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        """Build a result for a rate-limited request.

        Args:
            crossref_id: Cross-reference ID.
            document_id: Document ID.
            certificate_number: Certificate number.
            registry_type: Registry type.
            elapsed_ms: Processing time.

        Returns:
            Rate-limited result dictionary.
        """
        return {
            "success": False,
            "crossref_id": crossref_id,
            "document_id": document_id,
            "registry_type": registry_type,
            "certificate_number": certificate_number,
            "registry_found": False,
            "registry_status": None,
            "discrepancies": [
                f"Rate limit exceeded for {registry_type} registry; "
                f"retry after 60 seconds"
            ],
            "cached": False,
            "processing_time_ms": round(elapsed_ms, 2),
            "provenance_hash": _compute_hash({
                "crossref_id": crossref_id,
                "rate_limited": True,
            }),
            "error": "rate_limit_exceeded",
        }

    # ------------------------------------------------------------------
    # Internal: Name matching
    # ------------------------------------------------------------------

    @staticmethod
    def _names_match(
        claimed: str,
        registry: str,
    ) -> bool:
        """Compare two party names for a match.

        Uses case-insensitive containment and token overlap.

        Args:
            claimed: Claimed name.
            registry: Name from registry.

        Returns:
            True if considered a match.
        """
        if not claimed or not registry:
            return False

        c_lower = claimed.lower().strip()
        r_lower = registry.lower().strip()

        # Exact match
        if c_lower == r_lower:
            return True

        # Containment
        if c_lower in r_lower or r_lower in c_lower:
            return True

        # Token overlap (>= 50%)
        c_tokens = set(re.split(r"[\s,;./-]+", c_lower)) - {""}
        r_tokens = set(re.split(r"[\s,;./-]+", r_lower)) - {""}

        if not c_tokens or not r_tokens:
            return False

        overlap = len(c_tokens & r_tokens)
        max_tokens = max(len(c_tokens), len(r_tokens))
        return (overlap / max_tokens) >= 0.5

    # ------------------------------------------------------------------
    # Internal: Date parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(value: Any) -> Optional[datetime]:
        """Parse a date string to datetime.

        Args:
            value: Date string or datetime.

        Returns:
            Parsed datetime with UTC timezone or None.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        try:
            dt_str = str(value).strip()
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"):
                try:
                    parsed = datetime.strptime(dt_str, fmt)
                    return parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            # ISO format fallback
            if "T" in dt_str:
                parsed = datetime.fromisoformat(
                    dt_str.replace("Z", "+00:00"),
                )
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _ensure_datetime(value: Any) -> datetime:
        """Ensure a value is a datetime with UTC timezone.

        Args:
            value: datetime or parseable string.

        Returns:
            datetime with UTC timezone.
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        parsed = CrossReferenceVerifier._parse_date(value)
        if parsed:
            return parsed
        return utcnow()

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        with self._lock:
            cache_count = len(self._cache)
            result_count = len(self._results)
        return (
            f"CrossReferenceVerifier(cached={cache_count}, "
            f"results={result_count}, "
            f"ttl={self._config.cache_ttl_hours}h)"
        )

    def __len__(self) -> int:
        """Return the number of stored verification results."""
        with self._lock:
            return len(self._results)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CrossReferenceVerifier",
]
