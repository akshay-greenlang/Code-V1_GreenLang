# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-013: Blockchain Integration

18 Prometheus metrics for blockchain integration agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_bci_`` prefix (GreenLang EUDR Blockchain
Integration) for consistent identification in Prometheus queries, Grafana
dashboards, and alerting rules across the GreenLang platform.

Metrics (18 per PRD Section 7.3):
    Counters (13):
        1.  gl_eudr_bci_anchors_total                  - Anchor records created
        2.  gl_eudr_bci_anchors_confirmed_total        - Anchors confirmed on-chain
        3.  gl_eudr_bci_anchors_failed_total           - Anchor submissions failed
        4.  gl_eudr_bci_verifications_total            - Verification operations
        5.  gl_eudr_bci_verifications_tampered_total   - Tampered anchors detected
        6.  gl_eudr_bci_merkle_trees_total             - Merkle trees constructed
        7.  gl_eudr_bci_merkle_proofs_total            - Merkle proofs generated
        8.  gl_eudr_bci_events_indexed_total           - On-chain events indexed
        9.  gl_eudr_bci_contracts_deployed_total       - Smart contracts deployed
        10. gl_eudr_bci_access_grants_total            - Access grants issued
        11. gl_eudr_bci_evidence_packages_total        - Evidence packages created
        12. gl_eudr_bci_gas_spent_total                - Total gas spent (wei)
        13. gl_eudr_bci_api_errors_total               - API errors by operation

    Histograms (3):
        14. gl_eudr_bci_anchor_duration_seconds        - Anchor lifecycle latency
        15. gl_eudr_bci_verification_duration_seconds   - Verification latency
        16. gl_eudr_bci_merkle_build_duration_seconds   - Merkle tree build latency

    Gauges (2):
        17. gl_eudr_bci_active_listeners               - Active event listeners
        18. gl_eudr_bci_pending_anchors                - Pending anchor submissions

Label Values Reference:
    chain:
        ethereum, polygon, fabric, besu.
    anchor_event_type:
        dds_submission, custody_transfer, batch_event,
        certificate_reference, reconciliation_result,
        mass_balance_entry, document_authentication,
        geolocation_verification.
    priority:
        p0_immediate, p1_standard, p2_batch.
    contract_type:
        anchor_registry, custody_transfer, compliance_check.
    verification_status:
        verified, tampered, not_found, error.
    access_level:
        operator, competent_authority, auditor, supply_chain_partner.
    evidence_format:
        json, pdf, eudr_xml.
    operation:
        anchor, verify, build_tree, generate_proof, deploy_contract,
        index_event, grant_access, revoke_access, create_evidence,
        estimate_gas, batch_process.

Example:
    >>> from greenlang.agents.eudr.blockchain_integration.metrics import (
    ...     record_anchor_created,
    ...     record_anchor_confirmed,
    ...     record_verification,
    ...     observe_anchor_duration,
    ...     set_pending_anchors,
    ... )
    >>> record_anchor_created("polygon", "dds_submission")
    >>> record_anchor_confirmed("polygon")
    >>> record_verification("verified")
    >>> observe_anchor_duration(2.5)
    >>> set_pending_anchors(12)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013)
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; "
        "blockchain integration metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):  # type: ignore[return]
        """Create a Counter or retrieve existing one to avoid registry collisions."""
        try:
            return Counter(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Counter(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

    def _safe_histogram(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
        buckets: tuple = (),
    ):  # type: ignore[return]
        """Create a Histogram or retrieve existing one."""
        try:
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [], **kw,
            )
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(), **kw,
            )

    def _safe_gauge(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):  # type: ignore[return]
        """Create a Gauge or retrieve existing one."""
        try:
            return Gauge(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Gauge(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

# ---------------------------------------------------------------------------
# Metric definitions (18 metrics per PRD Section 7.3)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # -- Counters (13) -------------------------------------------------------

    # 1. Anchor records created by chain and event type
    bci_anchors_total = _safe_counter(
        "gl_eudr_bci_anchors_total",
        "Total anchor records created by chain and event type",
        labelnames=["chain", "anchor_event_type"],
    )

    # 2. Anchors confirmed on-chain by chain
    bci_anchors_confirmed_total = _safe_counter(
        "gl_eudr_bci_anchors_confirmed_total",
        "Total anchors confirmed on-chain by blockchain network",
        labelnames=["chain"],
    )

    # 3. Anchor submissions failed by chain
    bci_anchors_failed_total = _safe_counter(
        "gl_eudr_bci_anchors_failed_total",
        "Total anchor submissions that failed by blockchain network",
        labelnames=["chain"],
    )

    # 4. Verification operations by status
    bci_verifications_total = _safe_counter(
        "gl_eudr_bci_verifications_total",
        "Total verification operations by verification status",
        labelnames=["verification_status"],
    )

    # 5. Tampered anchors detected
    bci_verifications_tampered_total = _safe_counter(
        "gl_eudr_bci_verifications_tampered_total",
        "Total tampered anchors detected during verification",
    )

    # 6. Merkle trees constructed by hash algorithm
    bci_merkle_trees_total = _safe_counter(
        "gl_eudr_bci_merkle_trees_total",
        "Total Merkle trees constructed by hash algorithm",
        labelnames=["hash_algorithm"],
    )

    # 7. Merkle proofs generated
    bci_merkle_proofs_total = _safe_counter(
        "gl_eudr_bci_merkle_proofs_total",
        "Total Merkle proofs generated for anchor verification",
    )

    # 8. On-chain events indexed by event type
    bci_events_indexed_total = _safe_counter(
        "gl_eudr_bci_events_indexed_total",
        "Total on-chain events indexed by event type",
        labelnames=["event_type"],
    )

    # 9. Smart contracts deployed by contract type and chain
    bci_contracts_deployed_total = _safe_counter(
        "gl_eudr_bci_contracts_deployed_total",
        "Total smart contracts deployed by type and chain",
        labelnames=["contract_type", "chain"],
    )

    # 10. Access grants issued by access level
    bci_access_grants_total = _safe_counter(
        "gl_eudr_bci_access_grants_total",
        "Total access grants issued by access level",
        labelnames=["access_level"],
    )

    # 11. Evidence packages created by format
    bci_evidence_packages_total = _safe_counter(
        "gl_eudr_bci_evidence_packages_total",
        "Total evidence packages created by output format",
        labelnames=["evidence_format"],
    )

    # 12. Total gas spent (wei) by chain
    bci_gas_spent_total = _safe_counter(
        "gl_eudr_bci_gas_spent_total",
        "Total gas spent in wei by blockchain network",
        labelnames=["chain"],
    )

    # 13. API errors by operation
    bci_api_errors_total = _safe_counter(
        "gl_eudr_bci_api_errors_total",
        "Total API errors encountered by operation",
        labelnames=["operation"],
    )

    # -- Histograms (3) ------------------------------------------------------

    # 14. Anchor lifecycle latency (creation to confirmation)
    bci_anchor_duration_seconds = _safe_histogram(
        "gl_eudr_bci_anchor_duration_seconds",
        "Duration of anchor lifecycle from creation to confirmation in seconds",
        buckets=(
            0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
            60.0, 120.0, 300.0, 600.0, 1800.0,
        ),
    )

    # 15. Verification operation latency
    bci_verification_duration_seconds = _safe_histogram(
        "gl_eudr_bci_verification_duration_seconds",
        "Duration of verification operations in seconds",
        buckets=(
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
        ),
    )

    # 16. Merkle tree build latency
    bci_merkle_build_duration_seconds = _safe_histogram(
        "gl_eudr_bci_merkle_build_duration_seconds",
        "Duration of Merkle tree construction in seconds",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
        ),
    )

    # -- Gauges (2) ----------------------------------------------------------

    # 17. Active event listeners
    bci_active_listeners = _safe_gauge(
        "gl_eudr_bci_active_listeners",
        "Number of currently active on-chain event listeners",
    )

    # 18. Pending anchor submissions
    bci_pending_anchors = _safe_gauge(
        "gl_eudr_bci_pending_anchors",
        "Number of anchor submissions pending confirmation",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    bci_anchors_total = None                     # type: ignore[assignment]
    bci_anchors_confirmed_total = None           # type: ignore[assignment]
    bci_anchors_failed_total = None              # type: ignore[assignment]
    bci_verifications_total = None               # type: ignore[assignment]
    bci_verifications_tampered_total = None       # type: ignore[assignment]
    bci_merkle_trees_total = None                # type: ignore[assignment]
    bci_merkle_proofs_total = None               # type: ignore[assignment]
    bci_events_indexed_total = None              # type: ignore[assignment]
    bci_contracts_deployed_total = None          # type: ignore[assignment]
    bci_access_grants_total = None               # type: ignore[assignment]
    bci_evidence_packages_total = None           # type: ignore[assignment]
    bci_gas_spent_total = None                   # type: ignore[assignment]
    bci_api_errors_total = None                  # type: ignore[assignment]
    bci_anchor_duration_seconds = None           # type: ignore[assignment]
    bci_verification_duration_seconds = None     # type: ignore[assignment]
    bci_merkle_build_duration_seconds = None     # type: ignore[assignment]
    bci_active_listeners = None                  # type: ignore[assignment]
    bci_pending_anchors = None                   # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_anchor_created(chain: str, anchor_event_type: str) -> None:
    """Record an anchor creation event by chain and event type.

    Args:
        chain: Blockchain network (ethereum, polygon, fabric, besu).
        anchor_event_type: Type of EUDR event being anchored
            (dds_submission, custody_transfer, batch_event,
            certificate_reference, reconciliation_result,
            mass_balance_entry, document_authentication,
            geolocation_verification).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_anchors_total.labels(
        chain=chain, anchor_event_type=anchor_event_type,
    ).inc()


def record_anchor_confirmed(chain: str) -> None:
    """Record an anchor confirmation event by chain.

    Args:
        chain: Blockchain network where confirmation occurred.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_anchors_confirmed_total.labels(chain=chain).inc()


def record_anchor_failed(chain: str) -> None:
    """Record an anchor submission failure event by chain.

    Args:
        chain: Blockchain network where submission failed.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_anchors_failed_total.labels(chain=chain).inc()


def record_verification(verification_status: str) -> None:
    """Record a verification operation by status.

    Args:
        verification_status: Result status (verified, tampered,
            not_found, error).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_verifications_total.labels(
        verification_status=verification_status,
    ).inc()


def record_verification_tampered() -> None:
    """Record a tampered anchor detection event."""
    if not PROMETHEUS_AVAILABLE:
        return
    bci_verifications_tampered_total.inc()


def record_merkle_tree_built(hash_algorithm: str) -> None:
    """Record a Merkle tree construction event by hash algorithm.

    Args:
        hash_algorithm: Hash algorithm used (sha256, sha512, keccak256).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_merkle_trees_total.labels(
        hash_algorithm=hash_algorithm,
    ).inc()


def record_merkle_proof_generated() -> None:
    """Record a Merkle proof generation event."""
    if not PROMETHEUS_AVAILABLE:
        return
    bci_merkle_proofs_total.inc()


def record_event_indexed(event_type: str) -> None:
    """Record an on-chain event indexed event by event type.

    Args:
        event_type: Type of on-chain event (anchor_created,
            custody_transfer_recorded, compliance_check_completed,
            party_registered).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_events_indexed_total.labels(event_type=event_type).inc()


def record_contract_deployed(contract_type: str, chain: str) -> None:
    """Record a smart contract deployment event by type and chain.

    Args:
        contract_type: Type of contract (anchor_registry,
            custody_transfer, compliance_check).
        chain: Blockchain network where deployment occurred.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_contracts_deployed_total.labels(
        contract_type=contract_type, chain=chain,
    ).inc()


def record_access_grant(access_level: str) -> None:
    """Record an access grant issuance event by access level.

    Args:
        access_level: Grantee access level (operator,
            competent_authority, auditor, supply_chain_partner).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_access_grants_total.labels(access_level=access_level).inc()


def record_evidence_package(evidence_format: str) -> None:
    """Record an evidence package creation event by format.

    Args:
        evidence_format: Output format (json, pdf, eudr_xml).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_evidence_packages_total.labels(
        evidence_format=evidence_format,
    ).inc()


def record_gas_spent(chain: str, amount: float) -> None:
    """Record gas spent on a blockchain transaction.

    Args:
        chain: Blockchain network where gas was spent.
        amount: Gas amount in wei.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_gas_spent_total.labels(chain=chain).inc(amount)


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (anchor, verify,
            build_tree, generate_proof, deploy_contract,
            index_event, grant_access, revoke_access,
            create_evidence, estimate_gas, batch_process).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_api_errors_total.labels(operation=operation).inc()


def observe_anchor_duration(seconds: float) -> None:
    """Record the duration of an anchor lifecycle (creation to confirmation).

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_anchor_duration_seconds.observe(seconds)


def observe_verification_duration(seconds: float) -> None:
    """Record the duration of a verification operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_verification_duration_seconds.observe(seconds)


def observe_merkle_build_duration(seconds: float) -> None:
    """Record the duration of a Merkle tree construction.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_merkle_build_duration_seconds.observe(seconds)


def set_active_listeners(count: int) -> None:
    """Set the gauge for currently active on-chain event listeners.

    Args:
        count: Number of active event listeners. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_active_listeners.set(count)


def set_pending_anchors(count: int) -> None:
    """Set the gauge for pending anchor submissions.

    Args:
        count: Number of pending anchor submissions. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    bci_pending_anchors.set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "bci_anchors_total",
    "bci_anchors_confirmed_total",
    "bci_anchors_failed_total",
    "bci_verifications_total",
    "bci_verifications_tampered_total",
    "bci_merkle_trees_total",
    "bci_merkle_proofs_total",
    "bci_events_indexed_total",
    "bci_contracts_deployed_total",
    "bci_access_grants_total",
    "bci_evidence_packages_total",
    "bci_gas_spent_total",
    "bci_api_errors_total",
    "bci_anchor_duration_seconds",
    "bci_verification_duration_seconds",
    "bci_merkle_build_duration_seconds",
    "bci_active_listeners",
    "bci_pending_anchors",
    # Helper functions
    "record_anchor_created",
    "record_anchor_confirmed",
    "record_anchor_failed",
    "record_verification",
    "record_verification_tampered",
    "record_merkle_tree_built",
    "record_merkle_proof_generated",
    "record_event_indexed",
    "record_contract_deployed",
    "record_access_grant",
    "record_evidence_package",
    "record_gas_spent",
    "record_api_error",
    "observe_anchor_duration",
    "observe_verification_duration",
    "observe_merkle_build_duration",
    "set_active_listeners",
    "set_pending_anchors",
]
