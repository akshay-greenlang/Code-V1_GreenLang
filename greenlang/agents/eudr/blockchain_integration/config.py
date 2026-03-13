# -*- coding: utf-8 -*-
"""
Blockchain Integration Configuration - AGENT-EUDR-013

Centralized configuration for the Blockchain Integration Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Anchoring: batch size, interval, retry strategy, priority levels,
  gas price multiplier for on-chain EUDR compliance anchoring
- Smart contracts: gas limits, deploy timeouts, ABI caching for
  anchor registry, custody transfer, and compliance check contracts
- Multi-chain: primary/fallback chain selection, confirmation depths
  per network (Ethereum 12, Polygon 32, Fabric 1, Besu 1), RPC
  timeout, max connections per chain
- Verification: cache TTL, batch verify size, proof format for
  Merkle proof verification of anchored EUDR compliance data
- Event listener: polling interval, max events per poll, reorg depth,
  webhook timeout for on-chain event indexing
- Merkle tree: max leaves, sorted tree flag, hash algorithm for
  building tamper-evident Merkle trees over EUDR supply chain data
- Cross-party sharing: max grants per record, grant expiry, multi-party
  confirmation requirements for competent authority / auditor access
- Evidence: supported formats (JSON/PDF/EUDR_XML), retention years,
  package signing for Article 14 record-keeping compliance
- Batch processing: size limits, concurrency, timeout
- Data retention: EUDR Article 14 five-year retention
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_BCI_`` prefix (e.g. ``GL_EUDR_BCI_DATABASE_URL``,
``GL_EUDR_BCI_PRIMARY_CHAIN``).

Environment Variable Reference (GL_EUDR_BCI_ prefix):
    GL_EUDR_BCI_DATABASE_URL                   - PostgreSQL connection URL
    GL_EUDR_BCI_REDIS_URL                      - Redis connection URL
    GL_EUDR_BCI_LOG_LEVEL                      - Logging level
    GL_EUDR_BCI_POOL_SIZE                      - Database pool size
    GL_EUDR_BCI_BATCH_SIZE                     - Anchoring batch size
    GL_EUDR_BCI_BATCH_INTERVAL_S               - Anchoring batch interval seconds
    GL_EUDR_BCI_MAX_RETRIES                    - Max anchor submission retries
    GL_EUDR_BCI_RETRY_BACKOFF_FACTOR           - Retry backoff multiplier
    GL_EUDR_BCI_ANCHOR_PRIORITY_LEVELS         - Number of priority levels
    GL_EUDR_BCI_GAS_PRICE_MULTIPLIER           - Gas price bid multiplier
    GL_EUDR_BCI_DEFAULT_GAS_LIMIT              - Default smart contract gas limit
    GL_EUDR_BCI_CONTRACT_DEPLOY_TIMEOUT_S      - Contract deployment timeout
    GL_EUDR_BCI_ABI_CACHE_ENABLED              - Enable ABI cache
    GL_EUDR_BCI_PRIMARY_CHAIN                  - Primary blockchain network
    GL_EUDR_BCI_FALLBACK_CHAIN                 - Fallback blockchain network
    GL_EUDR_BCI_CONFIRMATION_DEPTH_ETHEREUM    - Ethereum confirmation depth
    GL_EUDR_BCI_CONFIRMATION_DEPTH_POLYGON     - Polygon confirmation depth
    GL_EUDR_BCI_CONFIRMATION_DEPTH_FABRIC      - Fabric confirmation depth
    GL_EUDR_BCI_CONFIRMATION_DEPTH_BESU        - Besu confirmation depth
    GL_EUDR_BCI_RPC_TIMEOUT_S                  - RPC call timeout seconds
    GL_EUDR_BCI_MAX_CONNECTIONS_PER_CHAIN      - Max RPC connections per chain
    GL_EUDR_BCI_VERIFICATION_CACHE_TTL_S       - Verification cache TTL seconds
    GL_EUDR_BCI_MAX_BATCH_VERIFY_SIZE          - Max batch verification size
    GL_EUDR_BCI_PROOF_FORMAT                   - Proof output format
    GL_EUDR_BCI_POLLING_INTERVAL_S             - Event listener polling interval
    GL_EUDR_BCI_MAX_EVENTS_PER_POLL            - Max events per poll cycle
    GL_EUDR_BCI_REORG_DEPTH                    - Chain reorganization depth
    GL_EUDR_BCI_WEBHOOK_TIMEOUT_S              - Webhook delivery timeout
    GL_EUDR_BCI_MAX_TREE_LEAVES                - Max Merkle tree leaves
    GL_EUDR_BCI_SORTED_TREE                    - Use sorted Merkle tree
    GL_EUDR_BCI_HASH_ALGORITHM                 - Merkle tree hash algorithm
    GL_EUDR_BCI_MAX_GRANTS_PER_RECORD          - Max access grants per record
    GL_EUDR_BCI_GRANT_EXPIRY_DAYS              - Access grant expiry days
    GL_EUDR_BCI_REQUIRE_MULTI_PARTY_CONFIRMATION - Require multi-party confirm
    GL_EUDR_BCI_MIN_CONFIRMATIONS              - Min multi-party confirmations
    GL_EUDR_BCI_EVIDENCE_FORMATS               - Supported evidence formats
    GL_EUDR_BCI_EVIDENCE_RETENTION_YEARS       - Evidence retention years
    GL_EUDR_BCI_PACKAGE_SIGNING_ENABLED        - Enable evidence package signing
    GL_EUDR_BCI_BATCH_MAX_SIZE                 - Batch processing max size
    GL_EUDR_BCI_BATCH_CONCURRENCY              - Batch concurrency workers
    GL_EUDR_BCI_BATCH_TIMEOUT_S                - Batch timeout seconds
    GL_EUDR_BCI_RETENTION_YEARS                - Data retention years
    GL_EUDR_BCI_ENABLE_PROVENANCE              - Enable provenance tracking
    GL_EUDR_BCI_GENESIS_HASH                   - Genesis hash anchor
    GL_EUDR_BCI_ENABLE_METRICS                 - Enable Prometheus metrics
    GL_EUDR_BCI_RATE_LIMIT                     - Max requests per minute

Example:
    >>> from greenlang.agents.eudr.blockchain_integration.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.primary_chain, cfg.confirmation_depth_ethereum)
    polygon 12

    >>> # Override for testing
    >>> from greenlang.agents.eudr.blockchain_integration.config import (
    ...     set_config, reset_config, BlockchainIntegrationConfig,
    ... )
    >>> set_config(BlockchainIntegrationConfig(primary_chain="ethereum"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_BCI_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid blockchain networks
# ---------------------------------------------------------------------------

_VALID_CHAINS = frozenset(
    {"ethereum", "polygon", "fabric", "besu"}
)

# ---------------------------------------------------------------------------
# Valid hash algorithms for Merkle tree construction
# ---------------------------------------------------------------------------

_VALID_HASH_ALGORITHMS = frozenset(
    {"sha256", "sha512", "keccak256"}
)

# ---------------------------------------------------------------------------
# Valid proof formats
# ---------------------------------------------------------------------------

_VALID_PROOF_FORMATS = frozenset(
    {"json", "binary"}
)

# ---------------------------------------------------------------------------
# Valid evidence formats
# ---------------------------------------------------------------------------

_VALID_EVIDENCE_FORMATS = frozenset(
    {"json", "pdf", "eudr_xml"}
)

# ---------------------------------------------------------------------------
# Valid contract types
# ---------------------------------------------------------------------------

_VALID_CONTRACT_TYPES = frozenset(
    {"anchor_registry", "custody_transfer", "compliance_check"}
)

# ---------------------------------------------------------------------------
# Valid anchor event types for EUDR supply chain
# ---------------------------------------------------------------------------

_VALID_ANCHOR_EVENT_TYPES = frozenset({
    "dds_submission",
    "custody_transfer",
    "batch_event",
    "certificate_reference",
    "reconciliation_result",
    "mass_balance_entry",
    "document_authentication",
    "geolocation_verification",
})

# ---------------------------------------------------------------------------
# Default confirmation depths per chain
# ---------------------------------------------------------------------------

_DEFAULT_CONFIRMATION_DEPTHS: Dict[str, int] = {
    "ethereum": 12,
    "polygon": 32,
    "fabric": 1,
    "besu": 1,
}

# ---------------------------------------------------------------------------
# Default evidence formats
# ---------------------------------------------------------------------------

_DEFAULT_EVIDENCE_FORMATS: List[str] = [
    "json", "pdf", "eudr_xml",
]

# ---------------------------------------------------------------------------
# Default EUDR commodities
# ---------------------------------------------------------------------------

_DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

# ---------------------------------------------------------------------------
# Default anchor priority levels (P0=immediate, P1=standard, P2=batch)
# ---------------------------------------------------------------------------

_DEFAULT_ANCHOR_PRIORITY_LEVELS: int = 3

# ---------------------------------------------------------------------------
# Default contract ABI directory (relative to deployment root)
# ---------------------------------------------------------------------------

_DEFAULT_ABI_DIRECTORY: str = "contracts/abi"

# ---------------------------------------------------------------------------
# Default RPC endpoints per chain (used in development only)
# ---------------------------------------------------------------------------

_DEFAULT_RPC_ENDPOINTS: Dict[str, str] = {
    "ethereum": "http://localhost:8545",
    "polygon": "http://localhost:8546",
    "fabric": "http://localhost:7051",
    "besu": "http://localhost:8547",
}


# ---------------------------------------------------------------------------
# BlockchainIntegrationConfig
# ---------------------------------------------------------------------------


@dataclass
class BlockchainIntegrationConfig:
    """Complete configuration for the EUDR Blockchain Integration Agent.

    Attributes are grouped by concern: connections, logging, anchoring,
    smart contracts, multi-chain, verification, event listener, Merkle
    tree, cross-party sharing, evidence packaging, batch processing,
    data retention, provenance tracking, metrics, and performance tuning.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_BCI_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage of
            anchor records, Merkle trees, access grants, and audit logs.
        redis_url: Redis connection URL for verification caching,
            rate limiting counters, and event deduplication.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        pool_size: PostgreSQL connection pool size.
        batch_size: Number of anchor events to accumulate before
            submitting as a single Merkle root to the blockchain.
        batch_interval_s: Maximum interval in seconds between anchor
            batch submissions regardless of batch fill level.
        max_retries: Maximum number of retries for failed anchor
            transaction submissions.
        retry_backoff_factor: Exponential backoff multiplier between
            retries (e.g. 2.0 means 2s, 4s, 8s).
        anchor_priority_levels: Number of anchor priority levels
            (P0=immediate, P1=standard, P2=batch).
        gas_price_multiplier: Multiplier applied to the estimated gas
            price to ensure timely transaction inclusion.
        default_gas_limit: Default gas limit for smart contract
            transactions.
        contract_deploy_timeout_s: Timeout in seconds for smart
            contract deployment transactions.
        abi_cache_enabled: Whether to cache compiled contract ABIs in
            memory to avoid repeated disk reads.
        primary_chain: Primary blockchain network for anchor
            submissions. One of: ethereum, polygon, fabric, besu.
        fallback_chain: Fallback blockchain network used when the
            primary chain is unavailable.
        confirmation_depth_ethereum: Number of block confirmations
            required on Ethereum before an anchor is considered final.
        confirmation_depth_polygon: Number of block confirmations
            required on Polygon before an anchor is considered final.
        confirmation_depth_fabric: Number of block confirmations
            required on Hyperledger Fabric (typically 1).
        confirmation_depth_besu: Number of block confirmations
            required on Hyperledger Besu (typically 1).
        rpc_timeout_s: Timeout in seconds for individual RPC calls
            to blockchain nodes.
        max_connections_per_chain: Maximum concurrent RPC connections
            maintained per blockchain network.
        verification_cache_ttl_s: TTL in seconds for verification
            result cache entries.
        max_batch_verify_size: Maximum number of records in a single
            batch verification request.
        proof_format: Output format for Merkle proofs (json or binary).
        polling_interval_s: Interval in seconds between event listener
            polling cycles for on-chain events.
        max_events_per_poll: Maximum number of events to process per
            polling cycle.
        reorg_depth: Number of blocks to look back when checking for
            chain reorganizations that may invalidate events.
        webhook_timeout_s: Timeout in seconds for delivering event
            notifications to registered webhooks.
        max_tree_leaves: Maximum number of leaves allowed in a single
            Merkle tree construction.
        sorted_tree: Whether to sort Merkle tree leaves before
            building the tree (recommended for deterministic proofs).
        hash_algorithm: Hash algorithm for Merkle tree construction.
            Supports sha256, sha512, keccak256.
        max_grants_per_record: Maximum number of access grants that
            can be issued for a single anchor record.
        grant_expiry_days: Default expiry in days for access grants
            issued to competent authorities and auditors.
        require_multi_party_confirmation: Whether to require multiple
            party confirmations before granting data access.
        min_confirmations: Minimum number of multi-party confirmations
            required when require_multi_party_confirmation is True.
        evidence_formats: Supported output formats for evidence
            packages (json, pdf, eudr_xml).
        evidence_retention_years: Number of years to retain evidence
            packages per EUDR Article 14.
        package_signing_enabled: Whether to digitally sign evidence
            packages for integrity verification.
        batch_max_size: Maximum number of records in a single batch
            processing job.
        batch_concurrency: Maximum concurrent batch processing workers.
        batch_timeout_s: Timeout in seconds for a single batch job.
        retention_years: Data retention in years per EUDR Article 14.
        eudr_commodities: List of EUDR-regulated commodity types.
        anchor_event_types: Supported anchor event types for EUDR
            supply chain operations.
        rpc_endpoints: Per-chain RPC endpoint URLs.
        confirmation_depths: Per-chain confirmation depth overrides.
        enable_provenance: Enable SHA-256 provenance chain tracking
            for all blockchain integration operations.
        genesis_hash: Genesis anchor string for the provenance chain,
            unique to the Blockchain Integration agent.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_bci_`` prefix.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10

    # -- Anchoring settings --------------------------------------------------
    batch_size: int = 100
    batch_interval_s: int = 300
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    anchor_priority_levels: int = 3
    gas_price_multiplier: float = 1.2

    # -- Smart contract settings ---------------------------------------------
    default_gas_limit: int = 200000
    contract_deploy_timeout_s: int = 120
    abi_cache_enabled: bool = True

    # -- Multi-chain settings ------------------------------------------------
    primary_chain: str = "polygon"
    fallback_chain: str = "ethereum"
    confirmation_depth_ethereum: int = 12
    confirmation_depth_polygon: int = 32
    confirmation_depth_fabric: int = 1
    confirmation_depth_besu: int = 1
    rpc_timeout_s: int = 30
    max_connections_per_chain: int = 5

    # -- Verification settings -----------------------------------------------
    verification_cache_ttl_s: int = 3600
    max_batch_verify_size: int = 1000
    proof_format: str = "json"

    # -- Event listener settings ---------------------------------------------
    polling_interval_s: int = 5
    max_events_per_poll: int = 100
    reorg_depth: int = 10
    webhook_timeout_s: int = 30

    # -- Merkle tree settings ------------------------------------------------
    max_tree_leaves: int = 10000
    sorted_tree: bool = True
    hash_algorithm: str = "sha256"

    # -- Cross-party sharing settings ----------------------------------------
    max_grants_per_record: int = 50
    grant_expiry_days: int = 365
    require_multi_party_confirmation: bool = True
    min_confirmations: int = 2

    # -- Evidence package settings -------------------------------------------
    evidence_formats: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EVIDENCE_FORMATS)
    )
    evidence_retention_years: int = 5
    package_signing_enabled: bool = True

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 500
    batch_concurrency: int = 4
    batch_timeout_s: int = 600

    # -- Data retention (EUDR Article 14) ------------------------------------
    retention_years: int = 5

    # -- EUDR commodities ----------------------------------------------------
    eudr_commodities: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EUDR_COMMODITIES)
    )

    # -- Anchor event types --------------------------------------------------
    anchor_event_types: List[str] = field(
        default_factory=lambda: sorted(_VALID_ANCHOR_EVENT_TYPES)
    )

    # -- RPC endpoints -------------------------------------------------------
    rpc_endpoints: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_RPC_ENDPOINTS)
    )

    # -- Confirmation depths -------------------------------------------------
    confirmation_depths: Dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_CONFIRMATION_DEPTHS)
    )

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-BCI-013-BLOCKCHAIN-INTEGRATION-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Rate limiting -------------------------------------------------------
    rate_limit: int = 200

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization.

        Performs range checks on all numeric fields, enumeration checks
        on string fields, threshold ordering validation, and
        normalization. Collects all errors before raising a single
        ValueError with all violations listed.

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a constraint.
        """
        errors: list[str] = []

        # -- Logging ---------------------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )
        else:
            self.log_level = normalised_log

        # -- Anchoring settings ----------------------------------------------
        if self.batch_size < 1:
            errors.append(
                f"batch_size must be >= 1, got {self.batch_size}"
            )
        if self.batch_size > 10000:
            errors.append(
                f"batch_size must be <= 10000, got {self.batch_size}"
            )

        if self.batch_interval_s < 1:
            errors.append(
                f"batch_interval_s must be >= 1, "
                f"got {self.batch_interval_s}"
            )
        if self.batch_interval_s > 86400:
            errors.append(
                f"batch_interval_s must be <= 86400 (24h), "
                f"got {self.batch_interval_s}"
            )

        if self.max_retries < 0:
            errors.append(
                f"max_retries must be >= 0, got {self.max_retries}"
            )
        if self.max_retries > 20:
            errors.append(
                f"max_retries must be <= 20, got {self.max_retries}"
            )

        if self.retry_backoff_factor < 1.0:
            errors.append(
                f"retry_backoff_factor must be >= 1.0, "
                f"got {self.retry_backoff_factor}"
            )
        if self.retry_backoff_factor > 10.0:
            errors.append(
                f"retry_backoff_factor must be <= 10.0, "
                f"got {self.retry_backoff_factor}"
            )

        if self.anchor_priority_levels < 1:
            errors.append(
                f"anchor_priority_levels must be >= 1, "
                f"got {self.anchor_priority_levels}"
            )
        if self.anchor_priority_levels > 10:
            errors.append(
                f"anchor_priority_levels must be <= 10, "
                f"got {self.anchor_priority_levels}"
            )

        if self.gas_price_multiplier < 1.0:
            errors.append(
                f"gas_price_multiplier must be >= 1.0, "
                f"got {self.gas_price_multiplier}"
            )
        if self.gas_price_multiplier > 10.0:
            errors.append(
                f"gas_price_multiplier must be <= 10.0, "
                f"got {self.gas_price_multiplier}"
            )

        # -- Smart contract settings -----------------------------------------
        if self.default_gas_limit < 21000:
            errors.append(
                f"default_gas_limit must be >= 21000 (minimum tx gas), "
                f"got {self.default_gas_limit}"
            )
        if self.default_gas_limit > 30000000:
            errors.append(
                f"default_gas_limit must be <= 30000000, "
                f"got {self.default_gas_limit}"
            )

        if self.contract_deploy_timeout_s < 10:
            errors.append(
                f"contract_deploy_timeout_s must be >= 10, "
                f"got {self.contract_deploy_timeout_s}"
            )
        if self.contract_deploy_timeout_s > 3600:
            errors.append(
                f"contract_deploy_timeout_s must be <= 3600 (1h), "
                f"got {self.contract_deploy_timeout_s}"
            )

        # -- Multi-chain settings --------------------------------------------
        normalised_primary = self.primary_chain.lower().strip()
        if normalised_primary not in _VALID_CHAINS:
            errors.append(
                f"primary_chain must be one of "
                f"{sorted(_VALID_CHAINS)}, "
                f"got '{self.primary_chain}'"
            )
        else:
            self.primary_chain = normalised_primary

        normalised_fallback = self.fallback_chain.lower().strip()
        if normalised_fallback not in _VALID_CHAINS:
            errors.append(
                f"fallback_chain must be one of "
                f"{sorted(_VALID_CHAINS)}, "
                f"got '{self.fallback_chain}'"
            )
        else:
            self.fallback_chain = normalised_fallback

        if self.confirmation_depth_ethereum < 1:
            errors.append(
                f"confirmation_depth_ethereum must be >= 1, "
                f"got {self.confirmation_depth_ethereum}"
            )
        if self.confirmation_depth_ethereum > 128:
            errors.append(
                f"confirmation_depth_ethereum must be <= 128, "
                f"got {self.confirmation_depth_ethereum}"
            )

        if self.confirmation_depth_polygon < 1:
            errors.append(
                f"confirmation_depth_polygon must be >= 1, "
                f"got {self.confirmation_depth_polygon}"
            )
        if self.confirmation_depth_polygon > 256:
            errors.append(
                f"confirmation_depth_polygon must be <= 256, "
                f"got {self.confirmation_depth_polygon}"
            )

        if self.confirmation_depth_fabric < 1:
            errors.append(
                f"confirmation_depth_fabric must be >= 1, "
                f"got {self.confirmation_depth_fabric}"
            )

        if self.confirmation_depth_besu < 1:
            errors.append(
                f"confirmation_depth_besu must be >= 1, "
                f"got {self.confirmation_depth_besu}"
            )

        if self.rpc_timeout_s < 1:
            errors.append(
                f"rpc_timeout_s must be >= 1, got {self.rpc_timeout_s}"
            )
        if self.rpc_timeout_s > 300:
            errors.append(
                f"rpc_timeout_s must be <= 300, "
                f"got {self.rpc_timeout_s}"
            )

        if self.max_connections_per_chain < 1:
            errors.append(
                f"max_connections_per_chain must be >= 1, "
                f"got {self.max_connections_per_chain}"
            )
        if self.max_connections_per_chain > 100:
            errors.append(
                f"max_connections_per_chain must be <= 100, "
                f"got {self.max_connections_per_chain}"
            )

        # -- Verification settings -------------------------------------------
        if self.verification_cache_ttl_s < 0:
            errors.append(
                f"verification_cache_ttl_s must be >= 0, "
                f"got {self.verification_cache_ttl_s}"
            )
        if self.verification_cache_ttl_s > 86400:
            errors.append(
                f"verification_cache_ttl_s must be <= 86400 (24h), "
                f"got {self.verification_cache_ttl_s}"
            )

        if self.max_batch_verify_size < 1:
            errors.append(
                f"max_batch_verify_size must be >= 1, "
                f"got {self.max_batch_verify_size}"
            )
        if self.max_batch_verify_size > 10000:
            errors.append(
                f"max_batch_verify_size must be <= 10000, "
                f"got {self.max_batch_verify_size}"
            )

        normalised_proof = self.proof_format.lower().strip()
        if normalised_proof not in _VALID_PROOF_FORMATS:
            errors.append(
                f"proof_format must be one of "
                f"{sorted(_VALID_PROOF_FORMATS)}, "
                f"got '{self.proof_format}'"
            )
        else:
            self.proof_format = normalised_proof

        # -- Event listener settings -----------------------------------------
        if self.polling_interval_s < 1:
            errors.append(
                f"polling_interval_s must be >= 1, "
                f"got {self.polling_interval_s}"
            )
        if self.polling_interval_s > 3600:
            errors.append(
                f"polling_interval_s must be <= 3600, "
                f"got {self.polling_interval_s}"
            )

        if self.max_events_per_poll < 1:
            errors.append(
                f"max_events_per_poll must be >= 1, "
                f"got {self.max_events_per_poll}"
            )
        if self.max_events_per_poll > 10000:
            errors.append(
                f"max_events_per_poll must be <= 10000, "
                f"got {self.max_events_per_poll}"
            )

        if self.reorg_depth < 0:
            errors.append(
                f"reorg_depth must be >= 0, got {self.reorg_depth}"
            )
        if self.reorg_depth > 128:
            errors.append(
                f"reorg_depth must be <= 128, got {self.reorg_depth}"
            )

        if self.webhook_timeout_s < 1:
            errors.append(
                f"webhook_timeout_s must be >= 1, "
                f"got {self.webhook_timeout_s}"
            )
        if self.webhook_timeout_s > 300:
            errors.append(
                f"webhook_timeout_s must be <= 300, "
                f"got {self.webhook_timeout_s}"
            )

        # -- Merkle tree settings --------------------------------------------
        if self.max_tree_leaves < 2:
            errors.append(
                f"max_tree_leaves must be >= 2, "
                f"got {self.max_tree_leaves}"
            )
        if self.max_tree_leaves > 1000000:
            errors.append(
                f"max_tree_leaves must be <= 1000000, "
                f"got {self.max_tree_leaves}"
            )

        normalised_hash = self.hash_algorithm.lower().strip()
        if normalised_hash not in _VALID_HASH_ALGORITHMS:
            errors.append(
                f"hash_algorithm must be one of "
                f"{sorted(_VALID_HASH_ALGORITHMS)}, "
                f"got '{self.hash_algorithm}'"
            )
        else:
            self.hash_algorithm = normalised_hash

        # -- Cross-party sharing settings ------------------------------------
        if self.max_grants_per_record < 1:
            errors.append(
                f"max_grants_per_record must be >= 1, "
                f"got {self.max_grants_per_record}"
            )
        if self.max_grants_per_record > 1000:
            errors.append(
                f"max_grants_per_record must be <= 1000, "
                f"got {self.max_grants_per_record}"
            )

        if self.grant_expiry_days < 1:
            errors.append(
                f"grant_expiry_days must be >= 1, "
                f"got {self.grant_expiry_days}"
            )
        if self.grant_expiry_days > 3650:
            errors.append(
                f"grant_expiry_days must be <= 3650 (10 years), "
                f"got {self.grant_expiry_days}"
            )

        if self.require_multi_party_confirmation:
            if self.min_confirmations < 1:
                errors.append(
                    f"min_confirmations must be >= 1 when "
                    f"require_multi_party_confirmation is True, "
                    f"got {self.min_confirmations}"
                )
            if self.min_confirmations > 10:
                errors.append(
                    f"min_confirmations must be <= 10, "
                    f"got {self.min_confirmations}"
                )

        # -- Evidence package settings ---------------------------------------
        if not self.evidence_formats:
            errors.append("evidence_formats must not be empty")
        for fmt in self.evidence_formats:
            if fmt.lower().strip() not in _VALID_EVIDENCE_FORMATS:
                errors.append(
                    f"evidence_formats contains invalid format "
                    f"'{fmt}'; valid: {sorted(_VALID_EVIDENCE_FORMATS)}"
                )

        if self.evidence_retention_years < 1:
            errors.append(
                f"evidence_retention_years must be >= 1, "
                f"got {self.evidence_retention_years}"
            )

        # -- Batch processing ------------------------------------------------
        if self.batch_max_size < 1:
            errors.append(
                f"batch_max_size must be >= 1, got {self.batch_max_size}"
            )

        if not (1 <= self.batch_concurrency <= 256):
            errors.append(
                f"batch_concurrency must be in [1, 256], "
                f"got {self.batch_concurrency}"
            )

        if self.batch_timeout_s < 1:
            errors.append(
                f"batch_timeout_s must be >= 1, got {self.batch_timeout_s}"
            )

        # -- Data retention --------------------------------------------------
        if self.retention_years < 1:
            errors.append(
                f"retention_years must be >= 1, "
                f"got {self.retention_years}"
            )

        # -- EUDR commodities ------------------------------------------------
        if not self.eudr_commodities:
            errors.append("eudr_commodities must not be empty")

        # -- Anchor event types ----------------------------------------------
        if not self.anchor_event_types:
            errors.append("anchor_event_types must not be empty")

        # -- RPC endpoints ---------------------------------------------------
        for chain, endpoint in self.rpc_endpoints.items():
            if chain.lower().strip() not in _VALID_CHAINS:
                errors.append(
                    f"rpc_endpoints contains unknown chain "
                    f"'{chain}'; valid: {sorted(_VALID_CHAINS)}"
                )
            if not endpoint:
                errors.append(
                    f"rpc_endpoints['{chain}'] must not be empty"
                )

        # -- Confirmation depths ---------------------------------------------
        for chain, depth in self.confirmation_depths.items():
            if depth < 1:
                errors.append(
                    f"confirmation_depths['{chain}'] must be >= 1, "
                    f"got {depth}"
                )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Performance tuning ----------------------------------------------
        if self.pool_size <= 0:
            errors.append(f"pool_size must be > 0, got {self.pool_size}")
        if self.rate_limit <= 0:
            errors.append(f"rate_limit must be > 0, got {self.rate_limit}")

        if errors:
            raise ValueError(
                "BlockchainIntegrationConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "BlockchainIntegrationConfig validated successfully: "
            "primary=%s, fallback=%s, batch_size=%d, "
            "batch_interval=%ds, max_retries=%d, "
            "gas_multiplier=%.1f, gas_limit=%d, "
            "confirm_eth=%d, confirm_poly=%d, "
            "confirm_fabric=%d, confirm_besu=%d, "
            "rpc_timeout=%ds, verify_cache=%ds, "
            "poll_interval=%ds, max_events=%d, reorg=%d, "
            "merkle_max=%d, sorted=%s, hash=%s, "
            "grants_max=%d, grant_expiry=%dd, multi_party=%s, "
            "evidence_formats=%s, evidence_retention=%dy, "
            "batch_max=%d, concurrency=%d, retention=%dy, "
            "provenance=%s, metrics=%s",
            self.primary_chain,
            self.fallback_chain,
            self.batch_size,
            self.batch_interval_s,
            self.max_retries,
            self.gas_price_multiplier,
            self.default_gas_limit,
            self.confirmation_depth_ethereum,
            self.confirmation_depth_polygon,
            self.confirmation_depth_fabric,
            self.confirmation_depth_besu,
            self.rpc_timeout_s,
            self.verification_cache_ttl_s,
            self.polling_interval_s,
            self.max_events_per_poll,
            self.reorg_depth,
            self.max_tree_leaves,
            self.sorted_tree,
            self.hash_algorithm,
            self.max_grants_per_record,
            self.grant_expiry_days,
            self.require_multi_party_confirmation,
            self.evidence_formats,
            self.evidence_retention_years,
            self.batch_max_size,
            self.batch_concurrency,
            self.retention_years,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> BlockchainIntegrationConfig:
        """Build a BlockchainIntegrationConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_BCI_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated BlockchainIntegrationConfig instance, validated via
            ``__post_init__``.
        """
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.strip().lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%r, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%r, using default %f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val.strip()

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            # Anchoring
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            batch_interval_s=_int(
                "BATCH_INTERVAL_S", cls.batch_interval_s,
            ),
            max_retries=_int("MAX_RETRIES", cls.max_retries),
            retry_backoff_factor=_float(
                "RETRY_BACKOFF_FACTOR", cls.retry_backoff_factor,
            ),
            anchor_priority_levels=_int(
                "ANCHOR_PRIORITY_LEVELS", cls.anchor_priority_levels,
            ),
            gas_price_multiplier=_float(
                "GAS_PRICE_MULTIPLIER", cls.gas_price_multiplier,
            ),
            # Smart contracts
            default_gas_limit=_int(
                "DEFAULT_GAS_LIMIT", cls.default_gas_limit,
            ),
            contract_deploy_timeout_s=_int(
                "CONTRACT_DEPLOY_TIMEOUT_S",
                cls.contract_deploy_timeout_s,
            ),
            abi_cache_enabled=_bool(
                "ABI_CACHE_ENABLED", cls.abi_cache_enabled,
            ),
            # Multi-chain
            primary_chain=_str(
                "PRIMARY_CHAIN", cls.primary_chain,
            ),
            fallback_chain=_str(
                "FALLBACK_CHAIN", cls.fallback_chain,
            ),
            confirmation_depth_ethereum=_int(
                "CONFIRMATION_DEPTH_ETHEREUM",
                cls.confirmation_depth_ethereum,
            ),
            confirmation_depth_polygon=_int(
                "CONFIRMATION_DEPTH_POLYGON",
                cls.confirmation_depth_polygon,
            ),
            confirmation_depth_fabric=_int(
                "CONFIRMATION_DEPTH_FABRIC",
                cls.confirmation_depth_fabric,
            ),
            confirmation_depth_besu=_int(
                "CONFIRMATION_DEPTH_BESU",
                cls.confirmation_depth_besu,
            ),
            rpc_timeout_s=_int(
                "RPC_TIMEOUT_S", cls.rpc_timeout_s,
            ),
            max_connections_per_chain=_int(
                "MAX_CONNECTIONS_PER_CHAIN",
                cls.max_connections_per_chain,
            ),
            # Verification
            verification_cache_ttl_s=_int(
                "VERIFICATION_CACHE_TTL_S",
                cls.verification_cache_ttl_s,
            ),
            max_batch_verify_size=_int(
                "MAX_BATCH_VERIFY_SIZE",
                cls.max_batch_verify_size,
            ),
            proof_format=_str(
                "PROOF_FORMAT", cls.proof_format,
            ),
            # Event listener
            polling_interval_s=_int(
                "POLLING_INTERVAL_S", cls.polling_interval_s,
            ),
            max_events_per_poll=_int(
                "MAX_EVENTS_PER_POLL", cls.max_events_per_poll,
            ),
            reorg_depth=_int("REORG_DEPTH", cls.reorg_depth),
            webhook_timeout_s=_int(
                "WEBHOOK_TIMEOUT_S", cls.webhook_timeout_s,
            ),
            # Merkle tree
            max_tree_leaves=_int(
                "MAX_TREE_LEAVES", cls.max_tree_leaves,
            ),
            sorted_tree=_bool("SORTED_TREE", cls.sorted_tree),
            hash_algorithm=_str(
                "HASH_ALGORITHM", cls.hash_algorithm,
            ),
            # Cross-party sharing
            max_grants_per_record=_int(
                "MAX_GRANTS_PER_RECORD",
                cls.max_grants_per_record,
            ),
            grant_expiry_days=_int(
                "GRANT_EXPIRY_DAYS", cls.grant_expiry_days,
            ),
            require_multi_party_confirmation=_bool(
                "REQUIRE_MULTI_PARTY_CONFIRMATION",
                cls.require_multi_party_confirmation,
            ),
            min_confirmations=_int(
                "MIN_CONFIRMATIONS", cls.min_confirmations,
            ),
            # Evidence
            evidence_retention_years=_int(
                "EVIDENCE_RETENTION_YEARS",
                cls.evidence_retention_years,
            ),
            package_signing_enabled=_bool(
                "PACKAGE_SIGNING_ENABLED",
                cls.package_signing_enabled,
            ),
            # Batch processing
            batch_max_size=_int(
                "BATCH_MAX_SIZE", cls.batch_max_size,
            ),
            batch_concurrency=_int(
                "BATCH_CONCURRENCY", cls.batch_concurrency,
            ),
            batch_timeout_s=_int(
                "BATCH_TIMEOUT_S", cls.batch_timeout_s,
            ),
            # Data retention
            retention_years=_int(
                "RETENTION_YEARS", cls.retention_years,
            ),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            # Rate limiting
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
        )

        logger.info(
            "BlockchainIntegrationConfig loaded: "
            "primary=%s, fallback=%s, "
            "batch_size=%d, batch_interval=%ds, "
            "max_retries=%d, backoff=%.1f, "
            "gas_multiplier=%.1f, gas_limit=%d, "
            "deploy_timeout=%ds, abi_cache=%s, "
            "confirm_eth=%d, confirm_poly=%d, "
            "confirm_fabric=%d, confirm_besu=%d, "
            "rpc_timeout=%ds, max_conn=%d, "
            "verify_cache=%ds, batch_verify=%d, "
            "proof=%s, poll=%ds, max_events=%d, "
            "reorg=%d, webhook=%ds, "
            "merkle_max=%d, sorted=%s, hash=%s, "
            "grants=%d, expiry=%dd, multi_party=%s, "
            "min_confirm=%d, "
            "evidence=%s, evidence_retention=%dy, "
            "signing=%s, "
            "batch_max=%d, concurrency=%d, timeout=%ds, "
            "retention=%dy, "
            "provenance=%s, pool=%d, rate_limit=%d/min, "
            "metrics=%s",
            config.primary_chain,
            config.fallback_chain,
            config.batch_size,
            config.batch_interval_s,
            config.max_retries,
            config.retry_backoff_factor,
            config.gas_price_multiplier,
            config.default_gas_limit,
            config.contract_deploy_timeout_s,
            config.abi_cache_enabled,
            config.confirmation_depth_ethereum,
            config.confirmation_depth_polygon,
            config.confirmation_depth_fabric,
            config.confirmation_depth_besu,
            config.rpc_timeout_s,
            config.max_connections_per_chain,
            config.verification_cache_ttl_s,
            config.max_batch_verify_size,
            config.proof_format,
            config.polling_interval_s,
            config.max_events_per_poll,
            config.reorg_depth,
            config.webhook_timeout_s,
            config.max_tree_leaves,
            config.sorted_tree,
            config.hash_algorithm,
            config.max_grants_per_record,
            config.grant_expiry_days,
            config.require_multi_party_confirmation,
            config.min_confirmations,
            config.evidence_formats,
            config.evidence_retention_years,
            config.package_signing_enabled,
            config.batch_max_size,
            config.batch_concurrency,
            config.batch_timeout_s,
            config.retention_years,
            config.enable_provenance,
            config.pool_size,
            config.rate_limit,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def anchoring_settings(self) -> Dict[str, Any]:
        """Return anchoring settings as a dictionary.

        Returns:
            Dictionary with keys: batch_size, batch_interval_s,
            max_retries, retry_backoff_factor, anchor_priority_levels,
            gas_price_multiplier.
        """
        return {
            "batch_size": self.batch_size,
            "batch_interval_s": self.batch_interval_s,
            "max_retries": self.max_retries,
            "retry_backoff_factor": self.retry_backoff_factor,
            "anchor_priority_levels": self.anchor_priority_levels,
            "gas_price_multiplier": self.gas_price_multiplier,
        }

    @property
    def contract_settings(self) -> Dict[str, Any]:
        """Return smart contract settings as a dictionary.

        Returns:
            Dictionary with keys: default_gas_limit,
            contract_deploy_timeout_s, abi_cache_enabled.
        """
        return {
            "default_gas_limit": self.default_gas_limit,
            "contract_deploy_timeout_s": self.contract_deploy_timeout_s,
            "abi_cache_enabled": self.abi_cache_enabled,
        }

    @property
    def chain_settings(self) -> Dict[str, Any]:
        """Return multi-chain settings as a dictionary.

        Returns:
            Dictionary with keys: primary_chain, fallback_chain,
            confirmation depths, rpc_timeout_s, max_connections_per_chain.
        """
        return {
            "primary_chain": self.primary_chain,
            "fallback_chain": self.fallback_chain,
            "confirmation_depth_ethereum": self.confirmation_depth_ethereum,
            "confirmation_depth_polygon": self.confirmation_depth_polygon,
            "confirmation_depth_fabric": self.confirmation_depth_fabric,
            "confirmation_depth_besu": self.confirmation_depth_besu,
            "rpc_timeout_s": self.rpc_timeout_s,
            "max_connections_per_chain": self.max_connections_per_chain,
        }

    @property
    def verification_settings(self) -> Dict[str, Any]:
        """Return verification settings as a dictionary.

        Returns:
            Dictionary with keys: verification_cache_ttl_s,
            max_batch_verify_size, proof_format.
        """
        return {
            "verification_cache_ttl_s": self.verification_cache_ttl_s,
            "max_batch_verify_size": self.max_batch_verify_size,
            "proof_format": self.proof_format,
        }

    @property
    def listener_settings(self) -> Dict[str, Any]:
        """Return event listener settings as a dictionary.

        Returns:
            Dictionary with keys: polling_interval_s,
            max_events_per_poll, reorg_depth, webhook_timeout_s.
        """
        return {
            "polling_interval_s": self.polling_interval_s,
            "max_events_per_poll": self.max_events_per_poll,
            "reorg_depth": self.reorg_depth,
            "webhook_timeout_s": self.webhook_timeout_s,
        }

    @property
    def merkle_settings(self) -> Dict[str, Any]:
        """Return Merkle tree settings as a dictionary.

        Returns:
            Dictionary with keys: max_tree_leaves, sorted_tree,
            hash_algorithm.
        """
        return {
            "max_tree_leaves": self.max_tree_leaves,
            "sorted_tree": self.sorted_tree,
            "hash_algorithm": self.hash_algorithm,
        }

    @property
    def sharing_settings(self) -> Dict[str, Any]:
        """Return cross-party sharing settings as a dictionary.

        Returns:
            Dictionary with keys: max_grants_per_record,
            grant_expiry_days, require_multi_party_confirmation,
            min_confirmations.
        """
        return {
            "max_grants_per_record": self.max_grants_per_record,
            "grant_expiry_days": self.grant_expiry_days,
            "require_multi_party_confirmation": (
                self.require_multi_party_confirmation
            ),
            "min_confirmations": self.min_confirmations,
        }

    @property
    def evidence_settings(self) -> Dict[str, Any]:
        """Return evidence package settings as a dictionary.

        Returns:
            Dictionary with keys: evidence_formats,
            evidence_retention_years, package_signing_enabled.
        """
        return {
            "evidence_formats": list(self.evidence_formats),
            "evidence_retention_years": self.evidence_retention_years,
            "package_signing_enabled": self.package_signing_enabled,
        }

    def get_confirmation_depth(self, chain: str) -> int:
        """Return the confirmation depth for a given blockchain network.

        Args:
            chain: Blockchain network identifier (ethereum, polygon,
                fabric, besu).

        Returns:
            Number of block confirmations required for finality.
        """
        chain_lower = chain.lower().strip()
        depth_map = {
            "ethereum": self.confirmation_depth_ethereum,
            "polygon": self.confirmation_depth_polygon,
            "fabric": self.confirmation_depth_fabric,
            "besu": self.confirmation_depth_besu,
        }
        return self.confirmation_depths.get(
            chain_lower,
            depth_map.get(chain_lower, 1),
        )

    def get_rpc_endpoint(self, chain: str) -> str:
        """Return the RPC endpoint URL for a given blockchain network.

        Args:
            chain: Blockchain network identifier (ethereum, polygon,
                fabric, besu).

        Returns:
            RPC endpoint URL string.
        """
        chain_lower = chain.lower().strip()
        return self.rpc_endpoints.get(
            chain_lower,
            _DEFAULT_RPC_ENDPOINTS.get(chain_lower, ""),
        )

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Re-run post-init validation and return True if valid.

        Returns:
            True if configuration passes all validation checks.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        self.__post_init__()
        return True

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain Python dictionary.

        Sensitive connection strings (database_url, redis_url) are
        redacted to prevent accidental credential leakage in logs,
        exception tracebacks, and monitoring dashboards.

        Returns:
            Dictionary representation with sensitive fields redacted.
        """
        return {
            # Connections (redacted)
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # Logging
            "log_level": self.log_level,
            # Performance tuning
            "pool_size": self.pool_size,
            # Anchoring
            "batch_size": self.batch_size,
            "batch_interval_s": self.batch_interval_s,
            "max_retries": self.max_retries,
            "retry_backoff_factor": self.retry_backoff_factor,
            "anchor_priority_levels": self.anchor_priority_levels,
            "gas_price_multiplier": self.gas_price_multiplier,
            # Smart contracts
            "default_gas_limit": self.default_gas_limit,
            "contract_deploy_timeout_s": self.contract_deploy_timeout_s,
            "abi_cache_enabled": self.abi_cache_enabled,
            # Multi-chain
            "primary_chain": self.primary_chain,
            "fallback_chain": self.fallback_chain,
            "confirmation_depth_ethereum": self.confirmation_depth_ethereum,
            "confirmation_depth_polygon": self.confirmation_depth_polygon,
            "confirmation_depth_fabric": self.confirmation_depth_fabric,
            "confirmation_depth_besu": self.confirmation_depth_besu,
            "rpc_timeout_s": self.rpc_timeout_s,
            "max_connections_per_chain": self.max_connections_per_chain,
            # Verification
            "verification_cache_ttl_s": self.verification_cache_ttl_s,
            "max_batch_verify_size": self.max_batch_verify_size,
            "proof_format": self.proof_format,
            # Event listener
            "polling_interval_s": self.polling_interval_s,
            "max_events_per_poll": self.max_events_per_poll,
            "reorg_depth": self.reorg_depth,
            "webhook_timeout_s": self.webhook_timeout_s,
            # Merkle tree
            "max_tree_leaves": self.max_tree_leaves,
            "sorted_tree": self.sorted_tree,
            "hash_algorithm": self.hash_algorithm,
            # Cross-party sharing
            "max_grants_per_record": self.max_grants_per_record,
            "grant_expiry_days": self.grant_expiry_days,
            "require_multi_party_confirmation": (
                self.require_multi_party_confirmation
            ),
            "min_confirmations": self.min_confirmations,
            # Evidence
            "evidence_formats": list(self.evidence_formats),
            "evidence_retention_years": self.evidence_retention_years,
            "package_signing_enabled": self.package_signing_enabled,
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            # Data retention
            "retention_years": self.retention_years,
            # EUDR commodities
            "eudr_commodities": list(self.eudr_commodities),
            # Anchor event types (count only)
            "anchor_event_types_count": len(self.anchor_event_types),
            # RPC endpoints (count only to avoid credential leakage)
            "rpc_endpoints_count": len(self.rpc_endpoints),
            # Confirmation depths
            "confirmation_depths": dict(self.confirmation_depths),
            # Provenance
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # Metrics
            "enable_metrics": self.enable_metrics,
            # Rate limiting
            "rate_limit": self.rate_limit,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Returns:
            String representation with sensitive fields redacted.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"BlockchainIntegrationConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[BlockchainIntegrationConfig] = None
_config_lock = threading.Lock()


def get_config() -> BlockchainIntegrationConfig:
    """Return the singleton BlockchainIntegrationConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_BCI_*`` environment variables.

    Returns:
        BlockchainIntegrationConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.primary_chain
        'polygon'
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = BlockchainIntegrationConfig.from_env()
    return _config_instance


def set_config(config: BlockchainIntegrationConfig) -> None:
    """Replace the singleton BlockchainIntegrationConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New BlockchainIntegrationConfig to install.

    Example:
        >>> cfg = BlockchainIntegrationConfig(primary_chain="ethereum")
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "BlockchainIntegrationConfig replaced programmatically: "
        "primary=%s, fallback=%s, batch_size=%d",
        config.primary_chain,
        config.fallback_chain,
        config.batch_size,
    )


def reset_config() -> None:
    """Reset the singleton BlockchainIntegrationConfig to None.

    The next call to get_config() will re-read GL_EUDR_BCI_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("BlockchainIntegrationConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "BlockchainIntegrationConfig",
    "get_config",
    "set_config",
    "reset_config",
]
