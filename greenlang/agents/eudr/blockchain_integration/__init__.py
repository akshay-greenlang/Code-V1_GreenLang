# -*- coding: utf-8 -*-
"""
Blockchain Integration Agent - AGENT-EUDR-013

Production-grade blockchain integration engine for EUDR compliance
covering on-chain anchoring of supply chain traceability data, Merkle
tree construction and proof generation, smart contract deployment and
lifecycle management, multi-chain connection handling, anchor
verification, on-chain event indexing, cross-party data access grant
management, evidence package generation with digital signing, gas cost
estimation and tracking, batch processing orchestration, and
comprehensive audit logging.

This package provides a complete blockchain integration system for
EUDR supply chain traceability supporting tamper-evident compliance
records per EU 2023/1115 Articles 4, 10, and 14:

    Capabilities:
        - On-chain anchoring of EUDR compliance data via Merkle root
          submission to Ethereum, Polygon, Hyperledger Fabric, and
          Hyperledger Besu with configurable confirmation depths
          (Ethereum 12, Polygon 32, Fabric 1, Besu 1)
        - Merkle tree construction from batched anchor events with
          SHA-256/SHA-512/Keccak-256 algorithms, sorted leaf ordering,
          and configurable max tree size (10,000 leaves)
        - Merkle inclusion proof generation for individual anchor
          verification against on-chain roots
        - Smart contract deployment and lifecycle management for
          anchor registry, custody transfer, and compliance check
          contracts with ABI caching and versioning
        - Multi-chain support with primary/fallback chain configuration,
          automatic failover, RPC connection pooling, and per-chain
          confirmation depth settings
        - Anchor verification via Merkle proof checking against
          on-chain roots with verification result caching
        - On-chain event indexing via configurable polling with reorg
          depth handling and webhook notification delivery
        - Cross-party data access grants for competent authorities,
          auditors, and supply chain partners with multi-party
          confirmation requirements and expiry management
        - Evidence package generation in JSON, PDF, and EUDR XML
          formats with digital signing for Article 14 record-keeping
        - Gas cost estimation and actual spend tracking per chain
          with gas price multiplier for timely inclusion
        - Batch processing for high-volume anchor operations with
          configurable concurrency and timeout
        - Priority-based anchor scheduling (P0 immediate, P1 standard,
          P2 batch) with retry and exponential backoff

    Foundational modules:
        - config: BlockchainIntegrationConfig with GL_EUDR_BCI_
          env var support (40+ settings)
        - models: Pydantic v2 data models with 15 enumerations,
          13 core models, 15 request models, and 15 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          12 entity types and 12 actions
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_bci_)

PRD: PRD-AGENT-EUDR-013
Agent ID: GL-EUDR-BCI-013
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.blockchain_integration import (
    ...     AnchorRecord,
    ...     AnchorEventType,
    ...     BlockchainNetwork,
    ...     AnchorStatus,
    ...     BlockchainIntegrationConfig,
    ...     get_config,
    ... )
    >>> anchor = AnchorRecord(
    ...     data_hash="a" * 64,
    ...     event_type=AnchorEventType.DDS_SUBMISSION,
    ...     operator_id="operator-001",
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-BCI-013"

# ---- Foundational: config ----
try:
    from greenlang.agents.eudr.blockchain_integration.config import (
        BlockchainIntegrationConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    BlockchainIntegrationConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---- Foundational: models ----
try:
    from greenlang.agents.eudr.blockchain_integration.models import (
        # Constants
        VERSION,
        EUDR_DEFORESTATION_CUTOFF,
        MAX_BATCH_SIZE,
        EUDR_RETENTION_YEARS,
        SUPPORTED_CHAINS,
        SUPPORTED_HASH_ALGORITHMS,
        SUPPORTED_ANCHOR_EVENT_TYPES,
        SUPPORTED_CONTRACT_TYPES,
        SUPPORTED_EVIDENCE_FORMATS,
        DEFAULT_CONFIRMATION_DEPTHS,
        # Enumerations
        BlockchainNetwork,
        AnchorStatus,
        AnchorEventType,
        AnchorPriority,
        ContractType,
        ContractStatus,
        VerificationStatus,
        EventType,
        AccessLevel,
        AccessStatus,
        EvidenceFormat,
        ProofFormat,
        TransactionStatus,
        ChainConnectionStatus,
        BatchJobStatus,
        # Core Models
        AnchorRecord,
        MerkleTree,
        MerkleLeaf,
        MerkleProof,
        SmartContract,
        ContractEvent,
        ChainConnection,
        VerificationResult,
        AccessGrant,
        EvidencePackage,
        GasCost,
        BatchJob,
        AuditLogEntry,
        # Request Models
        CreateAnchorRequest,
        BatchAnchorRequest,
        VerifyAnchorRequest,
        BuildMerkleTreeRequest,
        GenerateMerkleProofRequest,
        DeployContractRequest,
        SubmitTransactionRequest,
        StartListenerRequest,
        GrantAccessRequest,
        RevokeAccessRequest,
        CreateEvidenceRequest,
        EstimateGasRequest,
        SubmitBatchRequest,
        SearchAnchorsRequest,
        GetEventsRequest,
        # Response Models
        AnchorResponse,
        BatchAnchorResponse,
        VerificationResponse,
        MerkleTreeResponse,
        MerkleProofResponse,
        ContractResponse,
        TransactionResponse,
        ListenerResponse,
        AccessGrantResponse,
        AccessRevokeResponse,
        EvidenceResponse,
        GasEstimateResponse,
        BatchResponse,
        HealthResponse,
        DashboardResponse,
    )
except ImportError:
    pass

# ---- Foundational: provenance ----
try:
    from greenlang.agents.eudr.blockchain_integration.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_provenance_tracker,
        set_provenance_tracker,
        reset_provenance_tracker,
    )
except ImportError:
    ProvenanceRecord = None  # type: ignore[assignment,misc]
    ProvenanceTracker = None  # type: ignore[assignment,misc]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_provenance_tracker = None  # type: ignore[assignment]
    set_provenance_tracker = None  # type: ignore[assignment]
    reset_provenance_tracker = None  # type: ignore[assignment]

# ---- Foundational: metrics ----
try:
    from greenlang.agents.eudr.blockchain_integration.metrics import (
        PROMETHEUS_AVAILABLE,
        # Metric objects
        bci_anchors_total,
        bci_anchors_confirmed_total,
        bci_anchors_failed_total,
        bci_verifications_total,
        bci_verifications_tampered_total,
        bci_merkle_trees_total,
        bci_merkle_proofs_total,
        bci_events_indexed_total,
        bci_contracts_deployed_total,
        bci_access_grants_total,
        bci_evidence_packages_total,
        bci_gas_spent_total,
        bci_api_errors_total,
        bci_anchor_duration_seconds,
        bci_verification_duration_seconds,
        bci_merkle_build_duration_seconds,
        bci_active_listeners,
        bci_pending_anchors,
        # Helper functions
        record_anchor_created,
        record_anchor_confirmed,
        record_anchor_failed,
        record_verification,
        record_verification_tampered,
        record_merkle_tree_built,
        record_merkle_proof_generated,
        record_event_indexed,
        record_contract_deployed,
        record_access_grant,
        record_evidence_package,
        record_gas_spent,
        record_api_error,
        observe_anchor_duration,
        observe_verification_duration,
        observe_merkle_build_duration,
        set_active_listeners,
        set_pending_anchors,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]


__all__ = [
    # -- Version --
    "__version__",
    "__agent_id__",
    "VERSION",
    # -- Config --
    "BlockchainIntegrationConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    "SUPPORTED_CHAINS",
    "SUPPORTED_HASH_ALGORITHMS",
    "SUPPORTED_ANCHOR_EVENT_TYPES",
    "SUPPORTED_CONTRACT_TYPES",
    "SUPPORTED_EVIDENCE_FORMATS",
    "DEFAULT_CONFIRMATION_DEPTHS",
    # -- Enumerations --
    "BlockchainNetwork",
    "AnchorStatus",
    "AnchorEventType",
    "AnchorPriority",
    "ContractType",
    "ContractStatus",
    "VerificationStatus",
    "EventType",
    "AccessLevel",
    "AccessStatus",
    "EvidenceFormat",
    "ProofFormat",
    "TransactionStatus",
    "ChainConnectionStatus",
    "BatchJobStatus",
    # -- Core Models --
    "AnchorRecord",
    "MerkleTree",
    "MerkleLeaf",
    "MerkleProof",
    "SmartContract",
    "ContractEvent",
    "ChainConnection",
    "VerificationResult",
    "AccessGrant",
    "EvidencePackage",
    "GasCost",
    "BatchJob",
    "AuditLogEntry",
    # -- Request Models --
    "CreateAnchorRequest",
    "BatchAnchorRequest",
    "VerifyAnchorRequest",
    "BuildMerkleTreeRequest",
    "GenerateMerkleProofRequest",
    "DeployContractRequest",
    "SubmitTransactionRequest",
    "StartListenerRequest",
    "GrantAccessRequest",
    "RevokeAccessRequest",
    "CreateEvidenceRequest",
    "EstimateGasRequest",
    "SubmitBatchRequest",
    "SearchAnchorsRequest",
    "GetEventsRequest",
    # -- Response Models --
    "AnchorResponse",
    "BatchAnchorResponse",
    "VerificationResponse",
    "MerkleTreeResponse",
    "MerkleProofResponse",
    "ContractResponse",
    "TransactionResponse",
    "ListenerResponse",
    "AccessGrantResponse",
    "AccessRevokeResponse",
    "EvidenceResponse",
    "GasEstimateResponse",
    "BatchResponse",
    "HealthResponse",
    "DashboardResponse",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
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
