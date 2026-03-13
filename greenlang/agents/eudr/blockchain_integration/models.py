# -*- coding: utf-8 -*-
"""
Blockchain Integration Data Models - AGENT-EUDR-013

Pydantic v2 data models for the Blockchain Integration Agent covering
on-chain anchoring of EUDR compliance data, Merkle tree construction
and proof generation, smart contract deployment and lifecycle, multi-chain
connection management, anchor verification, on-chain event indexing,
cross-party data access grant management, evidence package generation,
gas cost estimation, batch processing orchestration, and audit logging.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all blockchain integration operations per
EU 2023/1115 Article 14.

Enumerations (15):
    - BlockchainNetwork, AnchorStatus, AnchorEventType, AnchorPriority,
      ContractType, ContractStatus, VerificationStatus, EventType,
      AccessLevel, AccessStatus, EvidenceFormat, ProofFormat,
      TransactionStatus, ChainConnectionStatus, BatchJobStatus

Core Models (12):
    - AnchorRecord, MerkleTree, MerkleLeaf, MerkleProof,
      SmartContract, ContractEvent, ChainConnection,
      VerificationResult, AccessGrant, EvidencePackage,
      GasCost, BatchJob, AuditLogEntry

Request Models (15):
    - CreateAnchorRequest, BatchAnchorRequest, VerifyAnchorRequest,
      BuildMerkleTreeRequest, GenerateMerkleProofRequest,
      DeployContractRequest, SubmitTransactionRequest,
      StartListenerRequest, GrantAccessRequest, RevokeAccessRequest,
      CreateEvidenceRequest, EstimateGasRequest, SubmitBatchRequest,
      SearchAnchorsRequest, GetEventsRequest

Response Models (15):
    - AnchorResponse, BatchAnchorResponse, VerificationResponse,
      MerkleTreeResponse, MerkleProofResponse, ContractResponse,
      TransactionResponse, ListenerResponse, AccessGrantResponse,
      AccessRevokeResponse, EvidenceResponse, GasEstimateResponse,
      BatchResponse, HealthResponse, DashboardResponse

Compatibility:
    Imports EUDRCommodity from greenlang.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector and AGENT-EUDR-011 Mass Balance Calculator.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Cross-agent commodity import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.eudr_traceability.models import EUDRCommodity
except ImportError:
    EUDRCommodity = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_DEFORESTATION_CUTOFF: str = "2020-12-31"

#: Maximum number of records in a single batch processing job.
MAX_BATCH_SIZE: int = 500

#: EUDR Article 14 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Supported blockchain networks.
SUPPORTED_CHAINS: List[str] = ["ethereum", "polygon", "fabric", "besu"]

#: Supported hash algorithms for Merkle tree construction.
SUPPORTED_HASH_ALGORITHMS: List[str] = ["sha256", "sha512", "keccak256"]

#: Supported anchor event types for EUDR supply chain.
SUPPORTED_ANCHOR_EVENT_TYPES: List[str] = [
    "dds_submission",
    "custody_transfer",
    "batch_event",
    "certificate_reference",
    "reconciliation_result",
    "mass_balance_entry",
    "document_authentication",
    "geolocation_verification",
]

#: Supported smart contract types.
SUPPORTED_CONTRACT_TYPES: List[str] = [
    "anchor_registry",
    "custody_transfer",
    "compliance_check",
]

#: Supported evidence package formats.
SUPPORTED_EVIDENCE_FORMATS: List[str] = ["json", "pdf", "eudr_xml"]

#: Default confirmation depths per chain.
DEFAULT_CONFIRMATION_DEPTHS: Dict[str, int] = {
    "ethereum": 12,
    "polygon": 32,
    "fabric": 1,
    "besu": 1,
}


# =============================================================================
# Enumerations
# =============================================================================


class BlockchainNetwork(str, Enum):
    """Supported blockchain networks for EUDR compliance anchoring.

    ETHEREUM: Ethereum mainnet or testnet (Sepolia/Goerli). Public
        permissionless chain with broad tooling support. Higher gas
        costs, slower confirmation (12 blocks ~3min).
    POLYGON: Polygon PoS network. EVM-compatible L2 with lower gas
        costs and faster confirmation (32 blocks ~64s). Primary
        chain for production EUDR anchoring.
    FABRIC: Hyperledger Fabric permissioned network. Enterprise-grade
        with channel-level privacy. Single block confirmation.
        Preferred for confidential supply chain data.
    BESU: Hyperledger Besu EVM-compatible enterprise chain. Supports
        IBFT 2.0 / QBFT consensus. Single block confirmation.
        Preferred for consortium deployments.
    """

    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    FABRIC = "fabric"
    BESU = "besu"


class AnchorStatus(str, Enum):
    """Status of an on-chain anchor record.

    PENDING: Anchor record created but not yet submitted to the
        blockchain. Awaiting batch aggregation or priority dispatch.
    SUBMITTED: Anchor transaction submitted to the blockchain network.
        Transaction hash assigned, awaiting mining.
    CONFIRMED: Anchor transaction mined and reached the required
        confirmation depth. Immutable on-chain record established.
    FAILED: Anchor submission failed due to transaction revert, gas
        exhaustion, network error, or retry limit exceeded.
    EXPIRED: Anchor record expired without confirmation. Occurs when
        a submitted transaction is not mined within the configured
        timeout window.
    """

    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    EXPIRED = "expired"


class AnchorEventType(str, Enum):
    """Type of EUDR supply chain event being anchored on-chain.

    DDS_SUBMISSION: Due Diligence Statement submission event per
        EUDR Article 4. Anchors the hash of a DDS for Article 14
        record-keeping compliance.
    CUSTODY_TRANSFER: Transfer of custody of regulated commodities
        between supply chain participants. Records sender, receiver,
        commodity, quantity, and geolocation.
    BATCH_EVENT: Batch-level processing event (splitting, merging,
        or transformation) affecting product traceability.
    CERTIFICATE_REFERENCE: Reference to an external certification
        (FSC, RSPO, ISCC, Fairtrade, UTZ/RA) with certificate ID
        and validity period.
    RECONCILIATION_RESULT: Mass balance reconciliation result between
        inbound and outbound commodity quantities.
    MASS_BALANCE_ENTRY: Individual mass balance ledger entry recording
        commodity inflow or outflow.
    DOCUMENT_AUTHENTICATION: Document authentication result from
        AGENT-EUDR-012 including hash integrity and signature status.
    GEOLOCATION_VERIFICATION: Geolocation verification result
        confirming production plot coordinates are deforestation-free.
    """

    DDS_SUBMISSION = "dds_submission"
    CUSTODY_TRANSFER = "custody_transfer"
    BATCH_EVENT = "batch_event"
    CERTIFICATE_REFERENCE = "certificate_reference"
    RECONCILIATION_RESULT = "reconciliation_result"
    MASS_BALANCE_ENTRY = "mass_balance_entry"
    DOCUMENT_AUTHENTICATION = "document_authentication"
    GEOLOCATION_VERIFICATION = "geolocation_verification"


class AnchorPriority(str, Enum):
    """Priority level for anchor submission scheduling.

    P0_IMMEDIATE: Immediate submission without batching. Used for
        critical compliance events (DDS submission, competent authority
        audit requests).
    P1_STANDARD: Standard priority with normal batch interval. Used
        for routine supply chain events (custody transfers, certificate
        references).
    P2_BATCH: Low priority deferred to batch submission. Used for
        high-volume events (mass balance entries, document
        authentication results).
    """

    P0_IMMEDIATE = "p0_immediate"
    P1_STANDARD = "p1_standard"
    P2_BATCH = "p2_batch"


class ContractType(str, Enum):
    """Type of smart contract deployed for EUDR compliance.

    ANCHOR_REGISTRY: Registry contract that stores Merkle roots of
        anchored EUDR compliance data. Provides verify() function
        for proof checking.
    CUSTODY_TRANSFER: Contract recording custody transfer events with
        sender/receiver validation and commodity type checking.
    COMPLIANCE_CHECK: Contract implementing on-chain compliance
        verification rules for EUDR Article 4 due diligence.
    """

    ANCHOR_REGISTRY = "anchor_registry"
    CUSTODY_TRANSFER = "custody_transfer"
    COMPLIANCE_CHECK = "compliance_check"


class ContractStatus(str, Enum):
    """Lifecycle status of a deployed smart contract.

    DEPLOYING: Contract deployment transaction submitted, awaiting
        mining and confirmation.
    DEPLOYED: Contract successfully deployed, address confirmed,
        ABI registered. Ready for interaction.
    PAUSED: Contract temporarily paused by administrator. No new
        transactions accepted, read operations still available.
    DEPRECATED: Contract deprecated in favor of a newer version.
        Historical data accessible, no new writes permitted.
    """

    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    PAUSED = "paused"
    DEPRECATED = "deprecated"


class VerificationStatus(str, Enum):
    """Result status of an anchor verification operation.

    VERIFIED: Anchor data matches the on-chain Merkle root. The
        supplied Merkle proof is valid and the data has not been
        tampered with.
    TAMPERED: Anchor data does NOT match the on-chain Merkle root.
        The data has been modified since anchoring.
    NOT_FOUND: Anchor record not found on-chain. The specified
        anchor ID or transaction hash does not exist in the registry.
    ERROR: Verification failed due to a system error (RPC timeout,
        contract call revert, invalid proof format).
    """

    VERIFIED = "verified"
    TAMPERED = "tampered"
    NOT_FOUND = "not_found"
    ERROR = "error"


class EventType(str, Enum):
    """Type of on-chain event emitted by EUDR smart contracts.

    ANCHOR_CREATED: Emitted when a new Merkle root is anchored in
        the registry contract.
    CUSTODY_TRANSFER_RECORDED: Emitted when a custody transfer event
        is recorded in the custody transfer contract.
    COMPLIANCE_CHECK_COMPLETED: Emitted when an on-chain compliance
        check is completed.
    PARTY_REGISTERED: Emitted when a new supply chain party is
        registered in the access control contract.
    """

    ANCHOR_CREATED = "anchor_created"
    CUSTODY_TRANSFER_RECORDED = "custody_transfer_recorded"
    COMPLIANCE_CHECK_COMPLETED = "compliance_check_completed"
    PARTY_REGISTERED = "party_registered"


class AccessLevel(str, Enum):
    """Access level for cross-party data sharing grants.

    OPERATOR: EUDR operator (importer/exporter) with full read/write
        access to their own supply chain data.
    COMPETENT_AUTHORITY: EU Member State competent authority with
        read-only access for Article 14 compliance verification.
    AUDITOR: Third-party auditor with read-only access scoped to
        specific audit engagement periods.
    SUPPLY_CHAIN_PARTNER: Upstream or downstream supply chain partner
        with read-only access to shared traceability data.
    """

    OPERATOR = "operator"
    COMPETENT_AUTHORITY = "competent_authority"
    AUDITOR = "auditor"
    SUPPLY_CHAIN_PARTNER = "supply_chain_partner"


class AccessStatus(str, Enum):
    """Status of a cross-party data access grant.

    ACTIVE: Grant is currently active and the grantee can access
        the specified anchor records.
    REVOKED: Grant has been explicitly revoked by the data owner.
        Access is immediately terminated.
    EXPIRED: Grant has passed its expiry date and access is no
        longer permitted. Requires re-issuance.
    """

    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class EvidenceFormat(str, Enum):
    """Output format for EUDR compliance evidence packages.

    JSON: Structured JSON format for machine-readable evidence
        suitable for API consumption and automated processing.
    PDF: Human-readable PDF report format suitable for competent
        authority submission and archival.
    EUDR_XML: EU Information System XML format as specified by the
        European Commission for EUDR Article 33 reporting.
    """

    JSON = "json"
    PDF = "pdf"
    EUDR_XML = "eudr_xml"


class ProofFormat(str, Enum):
    """Output format for Merkle proofs.

    JSON: JSON-encoded Merkle proof with sibling hashes and
        path indices for human-readable verification.
    BINARY: Compact binary-encoded Merkle proof for on-chain
        verification with minimal calldata.
    """

    JSON = "json"
    BINARY = "binary"


class TransactionStatus(str, Enum):
    """Status of a blockchain transaction.

    PENDING: Transaction submitted to the mempool, awaiting mining.
    MINED: Transaction included in a block but not yet at required
        confirmation depth.
    CONFIRMED: Transaction has reached the required confirmation
        depth and is considered final.
    REVERTED: Transaction was mined but execution reverted (e.g.,
        require() failure, out of gas).
    """

    PENDING = "pending"
    MINED = "mined"
    CONFIRMED = "confirmed"
    REVERTED = "reverted"


class ChainConnectionStatus(str, Enum):
    """Status of a blockchain network connection.

    CONNECTED: Successfully connected to the RPC endpoint with
        healthy block height and peer count.
    DISCONNECTED: Connection lost or not established. No RPC
        calls can be made.
    ERROR: Connection in error state due to authentication failure,
        rate limiting, or endpoint unavailability.
    RECONNECTING: Actively attempting to re-establish connection
        after a disconnection event.
    """

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class BatchJobStatus(str, Enum):
    """Status of a batch processing job.

    QUEUED: Job has been accepted and is waiting to be processed.
    PROCESSING: Job is currently being executed by a worker.
    COMPLETED: Job finished successfully with all records processed.
    FAILED: Job failed due to an unrecoverable error.
    CANCELLED: Job was cancelled by the operator before completion.
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Core Models
# =============================================================================


class AnchorRecord(BaseModel):
    """An on-chain anchor record for EUDR supply chain compliance data.

    Represents a single piece of EUDR compliance data (DDS submission,
    custody transfer, certificate reference, etc.) that has been hashed
    and anchored on a blockchain network via Merkle tree aggregation.

    Attributes:
        anchor_id: Unique anchor record identifier (UUID v4).
        data_hash: SHA-256 hash of the anchored compliance data payload.
        event_type: Type of EUDR supply chain event being anchored.
        chain: Blockchain network where the anchor is stored.
        status: Current anchor lifecycle status.
        priority: Submission priority level.
        merkle_root: Merkle root hash containing this anchor (set after
            tree construction).
        merkle_leaf_index: Position of this anchor's leaf in the Merkle
            tree (set after tree construction).
        tx_hash: Blockchain transaction hash (set after submission).
        block_number: Block number containing the transaction (set
            after mining).
        block_hash: Block hash containing the transaction.
        confirmations: Number of confirmations received so far.
        required_confirmations: Number of confirmations required for
            finality.
        gas_used: Actual gas consumed by the transaction (set after
            mining).
        gas_price_wei: Gas price in wei at submission time.
        operator_id: EUDR operator identifier who created the anchor.
        commodity: EUDR-regulated commodity type (optional).
        source_agent_id: Agent ID that produced the source data.
        source_record_id: Record ID in the source agent's domain.
        payload_metadata: Additional metadata about the anchored payload.
        retry_count: Number of submission retry attempts.
        error_message: Error message if anchor failed.
        created_at: UTC timestamp when the anchor was created.
        submitted_at: UTC timestamp when the transaction was submitted.
        confirmed_at: UTC timestamp when confirmation depth was reached.
        expires_at: UTC timestamp when the anchor record expires.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    anchor_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique anchor record identifier (UUID v4)",
    )
    data_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="SHA-256 hash of the anchored compliance data",
    )
    event_type: AnchorEventType = Field(
        ...,
        description="Type of EUDR supply chain event being anchored",
    )
    chain: BlockchainNetwork = Field(
        default=BlockchainNetwork.POLYGON,
        description="Blockchain network for this anchor",
    )
    status: AnchorStatus = Field(
        default=AnchorStatus.PENDING,
        description="Current anchor lifecycle status",
    )
    priority: AnchorPriority = Field(
        default=AnchorPriority.P1_STANDARD,
        description="Submission priority level",
    )
    merkle_root: Optional[str] = Field(
        None,
        description="Merkle root hash containing this anchor",
    )
    merkle_leaf_index: Optional[int] = Field(
        None,
        ge=0,
        description="Leaf index in the Merkle tree",
    )
    tx_hash: Optional[str] = Field(
        None,
        description="Blockchain transaction hash",
    )
    block_number: Optional[int] = Field(
        None,
        ge=0,
        description="Block number containing the transaction",
    )
    block_hash: Optional[str] = Field(
        None,
        description="Block hash containing the transaction",
    )
    confirmations: int = Field(
        default=0,
        ge=0,
        description="Number of confirmations received",
    )
    required_confirmations: int = Field(
        default=32,
        ge=1,
        description="Required confirmations for finality",
    )
    gas_used: Optional[int] = Field(
        None,
        ge=0,
        description="Actual gas consumed by the transaction",
    )
    gas_price_wei: Optional[int] = Field(
        None,
        ge=0,
        description="Gas price in wei at submission time",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )
    commodity: Optional[str] = Field(
        None,
        description="EUDR-regulated commodity type",
    )
    source_agent_id: Optional[str] = Field(
        None,
        description="Agent ID that produced the source data",
    )
    source_record_id: Optional[str] = Field(
        None,
        description="Record ID in the source agent's domain",
    )
    payload_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the anchored payload",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of submission retry attempts",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if anchor failed",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when anchor was created",
    )
    submitted_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when transaction was submitted",
    )
    confirmed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when confirmation depth was reached",
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when anchor record expires",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash for audit trail",
    )

    @field_validator("data_hash")
    @classmethod
    def validate_data_hash_hex(cls, v: str) -> str:
        """Validate data_hash is a valid hexadecimal string."""
        try:
            int(v, 16)
        except ValueError:
            raise ValueError(
                f"data_hash must be a valid hexadecimal string, got '{v}'"
            )
        return v.lower()


class MerkleLeaf(BaseModel):
    """A single leaf in a Merkle tree.

    Attributes:
        leaf_index: Position of this leaf in the tree (0-indexed).
        data_hash: SHA-256 hash of the leaf data.
        anchor_id: Associated anchor record identifier.
        leaf_hash: Computed leaf hash (may include prefix/domain separation).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    leaf_index: int = Field(
        ...,
        ge=0,
        description="Position of this leaf in the tree (0-indexed)",
    )
    data_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="SHA-256 hash of the leaf data",
    )
    anchor_id: str = Field(
        ...,
        min_length=1,
        description="Associated anchor record identifier",
    )
    leaf_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="Computed leaf hash",
    )


class MerkleTree(BaseModel):
    """A Merkle tree constructed from batched anchor records.

    Represents a complete Merkle tree built from a set of anchor data
    hashes. The root hash is anchored on-chain for tamper-evident
    verification of the entire batch.

    Attributes:
        tree_id: Unique Merkle tree identifier (UUID v4).
        root_hash: Computed Merkle root hash.
        leaf_count: Number of leaves in the tree.
        leaves: List of MerkleLeaf entries.
        depth: Tree depth (log2 of leaf count, rounded up).
        hash_algorithm: Hash algorithm used for tree construction.
        sorted: Whether leaves were sorted before tree construction.
        chain: Blockchain network where the root was anchored.
        tx_hash: Transaction hash of the on-chain anchor.
        block_number: Block number containing the anchor transaction.
        anchor_ids: List of anchor IDs included in this tree.
        created_at: UTC timestamp when the tree was constructed.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    tree_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique Merkle tree identifier (UUID v4)",
    )
    root_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="Computed Merkle root hash",
    )
    leaf_count: int = Field(
        ...,
        ge=1,
        description="Number of leaves in the tree",
    )
    leaves: List[MerkleLeaf] = Field(
        default_factory=list,
        description="List of MerkleLeaf entries",
    )
    depth: int = Field(
        ...,
        ge=0,
        description="Tree depth",
    )
    hash_algorithm: str = Field(
        default="sha256",
        description="Hash algorithm used for tree construction",
    )
    sorted: bool = Field(
        default=True,
        description="Whether leaves were sorted before construction",
    )
    chain: Optional[BlockchainNetwork] = Field(
        None,
        description="Blockchain network where root was anchored",
    )
    tx_hash: Optional[str] = Field(
        None,
        description="Transaction hash of the on-chain anchor",
    )
    block_number: Optional[int] = Field(
        None,
        ge=0,
        description="Block number containing the anchor transaction",
    )
    anchor_ids: List[str] = Field(
        default_factory=list,
        description="Anchor IDs included in this tree",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when tree was constructed",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )


class MerkleProof(BaseModel):
    """A Merkle inclusion proof for verifying a leaf against a root.

    Attributes:
        proof_id: Unique proof identifier (UUID v4).
        tree_id: Merkle tree identifier this proof belongs to.
        root_hash: Merkle root hash for verification target.
        leaf_hash: Hash of the leaf being proven.
        leaf_index: Index of the leaf in the tree.
        sibling_hashes: Ordered list of sibling hashes forming the
            authentication path from leaf to root.
        path_indices: List of 0/1 values indicating left(0) or
            right(1) position at each tree level.
        hash_algorithm: Hash algorithm used in tree construction.
        proof_format: Output format of this proof.
        anchor_id: Associated anchor record identifier.
        verified: Whether this proof has been locally verified.
        created_at: UTC timestamp when proof was generated.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    proof_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique proof identifier (UUID v4)",
    )
    tree_id: str = Field(
        ...,
        description="Merkle tree identifier",
    )
    root_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="Merkle root hash for verification",
    )
    leaf_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="Hash of the leaf being proven",
    )
    leaf_index: int = Field(
        ...,
        ge=0,
        description="Index of the leaf in the tree",
    )
    sibling_hashes: List[str] = Field(
        ...,
        description="Sibling hashes forming the authentication path",
    )
    path_indices: List[int] = Field(
        ...,
        description="Left(0)/right(1) indicators at each level",
    )
    hash_algorithm: str = Field(
        default="sha256",
        description="Hash algorithm used in tree construction",
    )
    proof_format: ProofFormat = Field(
        default=ProofFormat.JSON,
        description="Output format of this proof",
    )
    anchor_id: Optional[str] = Field(
        None,
        description="Associated anchor record identifier",
    )
    verified: Optional[bool] = Field(
        None,
        description="Whether proof has been locally verified",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when proof was generated",
    )

    @model_validator(mode="after")
    def validate_path_lengths_match(self) -> MerkleProof:
        """Validate sibling_hashes and path_indices have equal length."""
        if len(self.sibling_hashes) != len(self.path_indices):
            raise ValueError(
                f"sibling_hashes length ({len(self.sibling_hashes)}) "
                f"must equal path_indices length ({len(self.path_indices)})"
            )
        return self


class SmartContract(BaseModel):
    """A deployed smart contract for EUDR compliance operations.

    Attributes:
        contract_id: Unique contract identifier (UUID v4).
        contract_type: Type of EUDR smart contract.
        chain: Blockchain network where the contract is deployed.
        address: On-chain contract address (set after deployment).
        deployer_address: Address that deployed the contract.
        deploy_tx_hash: Deployment transaction hash.
        deploy_block_number: Block number of deployment transaction.
        abi_hash: SHA-256 hash of the contract ABI for versioning.
        version: Contract version string (semver).
        status: Current contract lifecycle status.
        created_at: UTC timestamp when contract record was created.
        deployed_at: UTC timestamp when contract was deployed.
        paused_at: UTC timestamp when contract was paused (if paused).
        deprecated_at: UTC timestamp when contract was deprecated.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    contract_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique contract identifier (UUID v4)",
    )
    contract_type: ContractType = Field(
        ...,
        description="Type of EUDR smart contract",
    )
    chain: BlockchainNetwork = Field(
        ...,
        description="Blockchain network for deployment",
    )
    address: Optional[str] = Field(
        None,
        description="On-chain contract address",
    )
    deployer_address: Optional[str] = Field(
        None,
        description="Address that deployed the contract",
    )
    deploy_tx_hash: Optional[str] = Field(
        None,
        description="Deployment transaction hash",
    )
    deploy_block_number: Optional[int] = Field(
        None,
        ge=0,
        description="Block number of deployment transaction",
    )
    abi_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the contract ABI",
    )
    version: str = Field(
        default="1.0.0",
        description="Contract version string (semver)",
    )
    status: ContractStatus = Field(
        default=ContractStatus.DEPLOYING,
        description="Current contract lifecycle status",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when record was created",
    )
    deployed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when contract was deployed",
    )
    paused_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when contract was paused",
    )
    deprecated_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when contract was deprecated",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )


class ContractEvent(BaseModel):
    """An on-chain event emitted by an EUDR smart contract.

    Attributes:
        event_id: Unique event identifier (UUID v4).
        event_type: Type of on-chain event.
        contract_address: Address of the emitting contract.
        chain: Blockchain network where the event was emitted.
        tx_hash: Transaction hash that emitted the event.
        block_number: Block number containing the event.
        block_hash: Block hash containing the event.
        log_index: Event log index within the transaction.
        event_data: Decoded event data payload.
        indexed_at: UTC timestamp when event was indexed.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier (UUID v4)",
    )
    event_type: EventType = Field(
        ...,
        description="Type of on-chain event",
    )
    contract_address: str = Field(
        ...,
        min_length=1,
        description="Address of the emitting contract",
    )
    chain: BlockchainNetwork = Field(
        ...,
        description="Blockchain network of the event",
    )
    tx_hash: str = Field(
        ...,
        min_length=1,
        description="Transaction hash that emitted the event",
    )
    block_number: int = Field(
        ...,
        ge=0,
        description="Block number containing the event",
    )
    block_hash: str = Field(
        ...,
        min_length=1,
        description="Block hash containing the event",
    )
    log_index: int = Field(
        ...,
        ge=0,
        description="Event log index within the transaction",
    )
    event_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Decoded event data payload",
    )
    indexed_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when event was indexed",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )


class ChainConnection(BaseModel):
    """A blockchain network connection descriptor.

    Attributes:
        connection_id: Unique connection identifier (UUID v4).
        chain: Blockchain network.
        rpc_endpoint: RPC endpoint URL (redacted in serialization).
        status: Current connection status.
        latest_block: Latest known block number on this chain.
        peer_count: Number of connected peers (for public chains).
        chain_id: Numeric chain ID (EVM chains).
        network_name: Human-readable network name.
        connected_at: UTC timestamp when connection was established.
        last_heartbeat: UTC timestamp of last successful RPC call.
        error_message: Error message if connection is in error state.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    connection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique connection identifier (UUID v4)",
    )
    chain: BlockchainNetwork = Field(
        ...,
        description="Blockchain network",
    )
    rpc_endpoint: str = Field(
        ...,
        min_length=1,
        description="RPC endpoint URL",
    )
    status: ChainConnectionStatus = Field(
        default=ChainConnectionStatus.DISCONNECTED,
        description="Current connection status",
    )
    latest_block: Optional[int] = Field(
        None,
        ge=0,
        description="Latest known block number",
    )
    peer_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of connected peers",
    )
    chain_id: Optional[int] = Field(
        None,
        description="Numeric chain ID (EVM chains)",
    )
    network_name: Optional[str] = Field(
        None,
        description="Human-readable network name",
    )
    connected_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when connection was established",
    )
    last_heartbeat: Optional[datetime] = Field(
        None,
        description="UTC timestamp of last successful RPC call",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if in error state",
    )


class VerificationResult(BaseModel):
    """Result of an anchor or Merkle proof verification operation.

    Attributes:
        verification_id: Unique verification identifier (UUID v4).
        anchor_id: Anchor record being verified.
        status: Verification result status.
        chain: Blockchain network used for verification.
        merkle_proof: Merkle proof used in verification (optional).
        on_chain_root: On-chain Merkle root retrieved from contract.
        computed_root: Locally computed Merkle root from proof.
        data_hash_match: Whether the data hash matches the anchor.
        root_hash_match: Whether computed root matches on-chain root.
        block_number: Block number at which verification was performed.
        gas_used: Gas consumed by on-chain verify() call (if any).
        cached: Whether this result was served from cache.
        error_message: Error message if verification failed.
        verified_at: UTC timestamp of verification.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    verification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique verification identifier (UUID v4)",
    )
    anchor_id: str = Field(
        ...,
        min_length=1,
        description="Anchor record being verified",
    )
    status: VerificationStatus = Field(
        ...,
        description="Verification result status",
    )
    chain: BlockchainNetwork = Field(
        ...,
        description="Blockchain network used for verification",
    )
    merkle_proof: Optional[MerkleProof] = Field(
        None,
        description="Merkle proof used in verification",
    )
    on_chain_root: Optional[str] = Field(
        None,
        description="On-chain Merkle root from contract",
    )
    computed_root: Optional[str] = Field(
        None,
        description="Locally computed Merkle root from proof",
    )
    data_hash_match: Optional[bool] = Field(
        None,
        description="Whether data hash matches the anchor",
    )
    root_hash_match: Optional[bool] = Field(
        None,
        description="Whether computed root matches on-chain root",
    )
    block_number: Optional[int] = Field(
        None,
        ge=0,
        description="Block number at verification time",
    )
    gas_used: Optional[int] = Field(
        None,
        ge=0,
        description="Gas consumed by on-chain verify() call",
    )
    cached: bool = Field(
        default=False,
        description="Whether result was served from cache",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if verification failed",
    )
    verified_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of verification",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )


class AccessGrant(BaseModel):
    """A cross-party data access grant for EUDR compliance sharing.

    Attributes:
        grant_id: Unique grant identifier (UUID v4).
        anchor_id: Anchor record being shared.
        grantor_id: Operator ID of the data owner (grantor).
        grantee_id: Identifier of the party receiving access.
        access_level: Level of access granted.
        status: Current grant status.
        scope: Optional scope restrictions (e.g., specific fields).
        multi_party_confirmations: Number of confirmations received.
        required_confirmations: Number of confirmations required.
        granted_at: UTC timestamp when grant was issued.
        expires_at: UTC timestamp when grant expires.
        revoked_at: UTC timestamp when grant was revoked (if revoked).
        revocation_reason: Reason for grant revocation.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    grant_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique grant identifier (UUID v4)",
    )
    anchor_id: str = Field(
        ...,
        min_length=1,
        description="Anchor record being shared",
    )
    grantor_id: str = Field(
        ...,
        min_length=1,
        description="Operator ID of the data owner",
    )
    grantee_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the receiving party",
    )
    access_level: AccessLevel = Field(
        ...,
        description="Level of access granted",
    )
    status: AccessStatus = Field(
        default=AccessStatus.ACTIVE,
        description="Current grant status",
    )
    scope: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional scope restrictions",
    )
    multi_party_confirmations: int = Field(
        default=0,
        ge=0,
        description="Number of confirmations received",
    )
    required_confirmations: int = Field(
        default=2,
        ge=1,
        description="Number of confirmations required",
    )
    granted_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when grant was issued",
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when grant expires",
    )
    revoked_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when grant was revoked",
    )
    revocation_reason: Optional[str] = Field(
        None,
        description="Reason for grant revocation",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )


class EvidencePackage(BaseModel):
    """An EUDR compliance evidence package containing blockchain proofs.

    Attributes:
        package_id: Unique package identifier (UUID v4).
        anchor_ids: List of anchor IDs included in this package.
        format: Output format of the evidence package.
        operator_id: EUDR operator identifier.
        merkle_proofs: Merkle proofs for each anchor in the package.
        verification_results: Verification results for each anchor.
        chain_references: On-chain references (tx hashes, block numbers).
        package_hash: SHA-256 hash of the complete package contents.
        signed: Whether the package has been digitally signed.
        signature: Digital signature of the package (if signed).
        signer_id: Identifier of the signing key (if signed).
        retention_until: UTC date until which the package must be
            retained per EUDR Article 14.
        created_at: UTC timestamp when package was created.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    package_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique package identifier (UUID v4)",
    )
    anchor_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Anchor IDs included in this package",
    )
    format: EvidenceFormat = Field(
        default=EvidenceFormat.JSON,
        description="Output format of the evidence package",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )
    merkle_proofs: List[MerkleProof] = Field(
        default_factory=list,
        description="Merkle proofs for each anchor",
    )
    verification_results: List[VerificationResult] = Field(
        default_factory=list,
        description="Verification results for each anchor",
    )
    chain_references: Dict[str, Any] = Field(
        default_factory=dict,
        description="On-chain references (tx hashes, block numbers)",
    )
    package_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of complete package contents",
    )
    signed: bool = Field(
        default=False,
        description="Whether package has been digitally signed",
    )
    signature: Optional[str] = Field(
        None,
        description="Digital signature of the package",
    )
    signer_id: Optional[str] = Field(
        None,
        description="Identifier of the signing key",
    )
    retention_until: Optional[datetime] = Field(
        None,
        description="Retention date per EUDR Article 14",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when package was created",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )


class GasCost(BaseModel):
    """Gas cost estimation or actual spend record for a blockchain transaction.

    Attributes:
        cost_id: Unique cost record identifier (UUID v4).
        chain: Blockchain network.
        operation: Type of operation (anchor, deploy, verify).
        estimated_gas: Estimated gas units for the operation.
        actual_gas: Actual gas consumed (set after mining).
        gas_price_wei: Gas price in wei.
        total_cost_wei: Total cost in wei (gas * price).
        total_cost_usd: Estimated USD equivalent (optional).
        tx_hash: Associated transaction hash (if applicable).
        created_at: UTC timestamp of estimation or spend.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    cost_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique cost record identifier (UUID v4)",
    )
    chain: BlockchainNetwork = Field(
        ...,
        description="Blockchain network",
    )
    operation: str = Field(
        ...,
        min_length=1,
        description="Type of operation",
    )
    estimated_gas: Optional[int] = Field(
        None,
        ge=0,
        description="Estimated gas units",
    )
    actual_gas: Optional[int] = Field(
        None,
        ge=0,
        description="Actual gas consumed",
    )
    gas_price_wei: Optional[int] = Field(
        None,
        ge=0,
        description="Gas price in wei",
    )
    total_cost_wei: Optional[int] = Field(
        None,
        ge=0,
        description="Total cost in wei",
    )
    total_cost_usd: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Estimated USD equivalent",
    )
    tx_hash: Optional[str] = Field(
        None,
        description="Associated transaction hash",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of estimation or spend",
    )


class BatchJob(BaseModel):
    """A batch processing job for bulk anchor operations.

    Attributes:
        job_id: Unique job identifier (UUID v4).
        status: Current job status.
        total_records: Total number of records to process.
        processed_records: Number of records processed so far.
        failed_records: Number of records that failed processing.
        anchor_ids: List of anchor IDs in this batch.
        chain: Target blockchain network.
        operator_id: EUDR operator who submitted the batch.
        error_message: Error message if job failed.
        started_at: UTC timestamp when processing started.
        completed_at: UTC timestamp when processing completed.
        created_at: UTC timestamp when job was created.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique job identifier (UUID v4)",
    )
    status: BatchJobStatus = Field(
        default=BatchJobStatus.QUEUED,
        description="Current job status",
    )
    total_records: int = Field(
        ...,
        ge=1,
        description="Total number of records to process",
    )
    processed_records: int = Field(
        default=0,
        ge=0,
        description="Number of records processed so far",
    )
    failed_records: int = Field(
        default=0,
        ge=0,
        description="Number of records that failed processing",
    )
    anchor_ids: List[str] = Field(
        default_factory=list,
        description="Anchor IDs in this batch",
    )
    chain: BlockchainNetwork = Field(
        default=BlockchainNetwork.POLYGON,
        description="Target blockchain network",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator who submitted the batch",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if job failed",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when processing started",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when processing completed",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when job was created",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )


class AuditLogEntry(BaseModel):
    """An audit log entry for blockchain integration operations.

    Attributes:
        entry_id: Unique entry identifier (UUID v4).
        entity_type: Type of entity involved.
        entity_id: Identifier of the entity involved.
        action: Action performed.
        actor_id: Identifier of the actor who performed the action.
        details: Additional details about the action.
        ip_address: IP address of the actor (if applicable).
        user_agent: User agent string (if applicable).
        created_at: UTC timestamp of the action.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    entry_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique entry identifier (UUID v4)",
    )
    entity_type: str = Field(
        ...,
        min_length=1,
        description="Type of entity involved",
    )
    entity_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the entity involved",
    )
    action: str = Field(
        ...,
        min_length=1,
        description="Action performed",
    )
    actor_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the actor",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the action",
    )
    ip_address: Optional[str] = Field(
        None,
        description="IP address of the actor",
    )
    user_agent: Optional[str] = Field(
        None,
        description="User agent string",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the action",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )


# =============================================================================
# Request Models
# =============================================================================


class CreateAnchorRequest(BaseModel):
    """Request to create a new anchor record.

    Attributes:
        data_hash: SHA-256 hash of the compliance data to anchor.
        event_type: Type of EUDR supply chain event.
        chain: Target blockchain network (default: polygon).
        priority: Submission priority level.
        operator_id: EUDR operator identifier.
        commodity: EUDR-regulated commodity type (optional).
        source_agent_id: Agent ID that produced the source data.
        source_record_id: Record ID in the source agent's domain.
        payload_metadata: Additional metadata about the payload.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    data_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="SHA-256 hash of the compliance data",
    )
    event_type: AnchorEventType = Field(
        ...,
        description="Type of EUDR supply chain event",
    )
    chain: BlockchainNetwork = Field(
        default=BlockchainNetwork.POLYGON,
        description="Target blockchain network",
    )
    priority: AnchorPriority = Field(
        default=AnchorPriority.P1_STANDARD,
        description="Submission priority level",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )
    commodity: Optional[str] = Field(
        None,
        description="EUDR-regulated commodity type",
    )
    source_agent_id: Optional[str] = Field(
        None,
        description="Agent ID that produced the source data",
    )
    source_record_id: Optional[str] = Field(
        None,
        description="Record ID in the source agent's domain",
    )
    payload_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the payload",
    )


class BatchAnchorRequest(BaseModel):
    """Request to create multiple anchor records in a batch.

    Attributes:
        anchors: List of individual anchor creation requests.
        chain: Target blockchain network for all anchors.
        operator_id: EUDR operator identifier.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    anchors: List[CreateAnchorRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of anchor creation requests",
    )
    chain: BlockchainNetwork = Field(
        default=BlockchainNetwork.POLYGON,
        description="Target blockchain network",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )


class VerifyAnchorRequest(BaseModel):
    """Request to verify an anchor record against on-chain data.

    Attributes:
        anchor_id: Anchor record identifier to verify.
        data_hash: Expected data hash for verification.
        chain: Blockchain network to verify against.
        include_proof: Whether to include the Merkle proof in response.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    anchor_id: str = Field(
        ...,
        min_length=1,
        description="Anchor record identifier to verify",
    )
    data_hash: Optional[str] = Field(
        None,
        description="Expected data hash for verification",
    )
    chain: Optional[BlockchainNetwork] = Field(
        None,
        description="Blockchain network to verify against",
    )
    include_proof: bool = Field(
        default=True,
        description="Include Merkle proof in response",
    )


class BuildMerkleTreeRequest(BaseModel):
    """Request to build a Merkle tree from anchor records.

    Attributes:
        anchor_ids: List of anchor IDs to include in the tree.
        hash_algorithm: Hash algorithm for tree construction.
        sorted: Whether to sort leaves before building.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    anchor_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Anchor IDs to include in the tree",
    )
    hash_algorithm: str = Field(
        default="sha256",
        description="Hash algorithm for tree construction",
    )
    sorted: bool = Field(
        default=True,
        description="Sort leaves before building",
    )


class GenerateMerkleProofRequest(BaseModel):
    """Request to generate a Merkle proof for an anchor.

    Attributes:
        tree_id: Merkle tree identifier.
        anchor_id: Anchor record for which to generate the proof.
        proof_format: Desired output format.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    tree_id: str = Field(
        ...,
        min_length=1,
        description="Merkle tree identifier",
    )
    anchor_id: str = Field(
        ...,
        min_length=1,
        description="Anchor record for proof generation",
    )
    proof_format: ProofFormat = Field(
        default=ProofFormat.JSON,
        description="Desired output format",
    )


class DeployContractRequest(BaseModel):
    """Request to deploy a smart contract.

    Attributes:
        contract_type: Type of EUDR smart contract to deploy.
        chain: Target blockchain network.
        deployer_address: Address deploying the contract.
        constructor_args: Constructor arguments (if any).
        gas_limit: Gas limit for deployment transaction.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    contract_type: ContractType = Field(
        ...,
        description="Type of EUDR smart contract to deploy",
    )
    chain: BlockchainNetwork = Field(
        ...,
        description="Target blockchain network",
    )
    deployer_address: str = Field(
        ...,
        min_length=1,
        description="Address deploying the contract",
    )
    constructor_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor arguments",
    )
    gas_limit: Optional[int] = Field(
        None,
        ge=21000,
        description="Gas limit for deployment",
    )


class SubmitTransactionRequest(BaseModel):
    """Request to submit a raw transaction to the blockchain.

    Attributes:
        chain: Target blockchain network.
        contract_address: Target contract address.
        function_name: Contract function to call.
        function_args: Function call arguments.
        sender_address: Address sending the transaction.
        gas_limit: Gas limit for the transaction.
        value_wei: ETH/MATIC value to send (in wei).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    chain: BlockchainNetwork = Field(
        ...,
        description="Target blockchain network",
    )
    contract_address: str = Field(
        ...,
        min_length=1,
        description="Target contract address",
    )
    function_name: str = Field(
        ...,
        min_length=1,
        description="Contract function to call",
    )
    function_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Function call arguments",
    )
    sender_address: str = Field(
        ...,
        min_length=1,
        description="Address sending the transaction",
    )
    gas_limit: Optional[int] = Field(
        None,
        ge=21000,
        description="Gas limit for the transaction",
    )
    value_wei: int = Field(
        default=0,
        ge=0,
        description="ETH/MATIC value to send (in wei)",
    )


class StartListenerRequest(BaseModel):
    """Request to start an on-chain event listener.

    Attributes:
        chain: Blockchain network to listen on.
        contract_address: Contract address to monitor.
        event_types: Event types to listen for.
        from_block: Block number to start listening from.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    chain: BlockchainNetwork = Field(
        ...,
        description="Blockchain network to listen on",
    )
    contract_address: str = Field(
        ...,
        min_length=1,
        description="Contract address to monitor",
    )
    event_types: List[EventType] = Field(
        default_factory=list,
        description="Event types to listen for",
    )
    from_block: Optional[int] = Field(
        None,
        ge=0,
        description="Block number to start listening from",
    )


class GrantAccessRequest(BaseModel):
    """Request to grant cross-party access to anchor data.

    Attributes:
        anchor_id: Anchor record to share.
        grantee_id: Identifier of the receiving party.
        access_level: Level of access to grant.
        scope: Optional scope restrictions.
        expiry_days: Number of days until grant expires.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    anchor_id: str = Field(
        ...,
        min_length=1,
        description="Anchor record to share",
    )
    grantee_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the receiving party",
    )
    access_level: AccessLevel = Field(
        ...,
        description="Level of access to grant",
    )
    scope: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional scope restrictions",
    )
    expiry_days: int = Field(
        default=365,
        ge=1,
        le=3650,
        description="Number of days until grant expires",
    )


class RevokeAccessRequest(BaseModel):
    """Request to revoke a previously issued access grant.

    Attributes:
        grant_id: Grant identifier to revoke.
        reason: Reason for revocation.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    grant_id: str = Field(
        ...,
        min_length=1,
        description="Grant identifier to revoke",
    )
    reason: Optional[str] = Field(
        None,
        description="Reason for revocation",
    )


class CreateEvidenceRequest(BaseModel):
    """Request to create an evidence package.

    Attributes:
        anchor_ids: Anchor IDs to include in the package.
        format: Desired output format.
        operator_id: EUDR operator identifier.
        include_proofs: Whether to include Merkle proofs.
        include_verification: Whether to include verification results.
        sign_package: Whether to digitally sign the package.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    anchor_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Anchor IDs to include in the package",
    )
    format: EvidenceFormat = Field(
        default=EvidenceFormat.JSON,
        description="Desired output format",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )
    include_proofs: bool = Field(
        default=True,
        description="Include Merkle proofs",
    )
    include_verification: bool = Field(
        default=True,
        description="Include verification results",
    )
    sign_package: bool = Field(
        default=True,
        description="Digitally sign the package",
    )


class EstimateGasRequest(BaseModel):
    """Request to estimate gas for a blockchain operation.

    Attributes:
        chain: Target blockchain network.
        operation: Type of operation to estimate.
        parameters: Operation-specific parameters.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    chain: BlockchainNetwork = Field(
        ...,
        description="Target blockchain network",
    )
    operation: str = Field(
        ...,
        min_length=1,
        description="Type of operation to estimate",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation-specific parameters",
    )


class SubmitBatchRequest(BaseModel):
    """Request to submit a batch processing job.

    Attributes:
        anchor_ids: Anchor IDs to process in the batch.
        chain: Target blockchain network.
        operator_id: EUDR operator identifier.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    anchor_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Anchor IDs to process in the batch",
    )
    chain: BlockchainNetwork = Field(
        default=BlockchainNetwork.POLYGON,
        description="Target blockchain network",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )


class SearchAnchorsRequest(BaseModel):
    """Request to search anchor records with filters.

    Attributes:
        operator_id: Filter by EUDR operator identifier.
        event_type: Filter by anchor event type.
        chain: Filter by blockchain network.
        status: Filter by anchor status.
        commodity: Filter by EUDR commodity type.
        date_from: Filter anchors created after this date.
        date_to: Filter anchors created before this date.
        limit: Maximum number of results to return.
        offset: Offset for pagination.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    operator_id: Optional[str] = Field(
        None,
        description="Filter by EUDR operator identifier",
    )
    event_type: Optional[AnchorEventType] = Field(
        None,
        description="Filter by anchor event type",
    )
    chain: Optional[BlockchainNetwork] = Field(
        None,
        description="Filter by blockchain network",
    )
    status: Optional[AnchorStatus] = Field(
        None,
        description="Filter by anchor status",
    )
    commodity: Optional[str] = Field(
        None,
        description="Filter by EUDR commodity type",
    )
    date_from: Optional[datetime] = Field(
        None,
        description="Filter anchors created after this date",
    )
    date_to: Optional[datetime] = Field(
        None,
        description="Filter anchors created before this date",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of results",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Offset for pagination",
    )


class GetEventsRequest(BaseModel):
    """Request to retrieve indexed on-chain events.

    Attributes:
        chain: Filter by blockchain network.
        event_type: Filter by event type.
        contract_address: Filter by contract address.
        from_block: Filter events from this block number.
        to_block: Filter events up to this block number.
        limit: Maximum number of results to return.
        offset: Offset for pagination.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    chain: Optional[BlockchainNetwork] = Field(
        None,
        description="Filter by blockchain network",
    )
    event_type: Optional[EventType] = Field(
        None,
        description="Filter by event type",
    )
    contract_address: Optional[str] = Field(
        None,
        description="Filter by contract address",
    )
    from_block: Optional[int] = Field(
        None,
        ge=0,
        description="Filter events from this block number",
    )
    to_block: Optional[int] = Field(
        None,
        ge=0,
        description="Filter events up to this block number",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of results",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Offset for pagination",
    )


# =============================================================================
# Response Models
# =============================================================================


class AnchorResponse(BaseModel):
    """Response for a single anchor operation.

    Attributes:
        success: Whether the operation succeeded.
        anchor: The anchor record (if successful).
        error: Error message (if failed).
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    anchor: Optional[AnchorRecord] = Field(
        None, description="Anchor record",
    )
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )
    provenance_hash: Optional[str] = Field(
        None, description="SHA-256 provenance chain hash",
    )


class BatchAnchorResponse(BaseModel):
    """Response for a batch anchor creation operation.

    Attributes:
        success: Whether the batch operation succeeded.
        total: Total number of anchors in the batch.
        created: Number of anchors successfully created.
        failed: Number of anchors that failed creation.
        anchors: List of created anchor records.
        errors: List of error messages for failed anchors.
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Batch operation success flag")
    total: int = Field(..., ge=0, description="Total anchors in batch")
    created: int = Field(default=0, ge=0, description="Anchors created")
    failed: int = Field(default=0, ge=0, description="Anchors failed")
    anchors: List[AnchorRecord] = Field(
        default_factory=list, description="Created anchor records",
    )
    errors: List[str] = Field(
        default_factory=list, description="Error messages for failures",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )


class VerificationResponse(BaseModel):
    """Response for an anchor verification operation.

    Attributes:
        success: Whether the verification completed without errors.
        result: The verification result.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    result: Optional[VerificationResult] = Field(
        None, description="Verification result",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )
    provenance_hash: Optional[str] = Field(
        None, description="SHA-256 provenance chain hash",
    )


class MerkleTreeResponse(BaseModel):
    """Response for a Merkle tree construction operation.

    Attributes:
        success: Whether the tree was constructed successfully.
        tree: The constructed Merkle tree.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    tree: Optional[MerkleTree] = Field(
        None, description="Constructed Merkle tree",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )
    provenance_hash: Optional[str] = Field(
        None, description="SHA-256 provenance chain hash",
    )


class MerkleProofResponse(BaseModel):
    """Response for a Merkle proof generation operation.

    Attributes:
        success: Whether the proof was generated successfully.
        proof: The generated Merkle proof.
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    proof: Optional[MerkleProof] = Field(
        None, description="Generated Merkle proof",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )


class ContractResponse(BaseModel):
    """Response for a smart contract operation.

    Attributes:
        success: Whether the operation succeeded.
        contract: The smart contract record.
        error: Error message (if failed).
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    contract: Optional[SmartContract] = Field(
        None, description="Smart contract record",
    )
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )


class TransactionResponse(BaseModel):
    """Response for a blockchain transaction submission.

    Attributes:
        success: Whether the transaction was submitted.
        tx_hash: Transaction hash.
        status: Transaction status.
        block_number: Block number (if mined).
        gas_used: Gas consumed (if mined).
        error: Error message (if failed).
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    tx_hash: Optional[str] = Field(
        None, description="Transaction hash",
    )
    status: Optional[TransactionStatus] = Field(
        None, description="Transaction status",
    )
    block_number: Optional[int] = Field(
        None, ge=0, description="Block number",
    )
    gas_used: Optional[int] = Field(
        None, ge=0, description="Gas consumed",
    )
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )


class ListenerResponse(BaseModel):
    """Response for an event listener operation.

    Attributes:
        success: Whether the listener was started/stopped.
        chain: Blockchain network being listened on.
        contract_address: Contract address being monitored.
        active: Whether the listener is currently active.
        last_block: Last processed block number.
        error: Error message (if failed).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    chain: Optional[BlockchainNetwork] = Field(
        None, description="Blockchain network",
    )
    contract_address: Optional[str] = Field(
        None, description="Contract address",
    )
    active: bool = Field(
        default=False, description="Listener active status",
    )
    last_block: Optional[int] = Field(
        None, ge=0, description="Last processed block",
    )
    error: Optional[str] = Field(None, description="Error message")


class AccessGrantResponse(BaseModel):
    """Response for an access grant operation.

    Attributes:
        success: Whether the grant was issued.
        grant: The access grant record.
        error: Error message (if failed).
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    grant: Optional[AccessGrant] = Field(
        None, description="Access grant record",
    )
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )


class AccessRevokeResponse(BaseModel):
    """Response for an access revocation operation.

    Attributes:
        success: Whether the revocation succeeded.
        grant_id: Identifier of the revoked grant.
        error: Error message (if failed).
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    grant_id: Optional[str] = Field(
        None, description="Revoked grant identifier",
    )
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )


class EvidenceResponse(BaseModel):
    """Response for an evidence package creation operation.

    Attributes:
        success: Whether the package was created.
        package: The evidence package record.
        error: Error message (if failed).
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    package: Optional[EvidencePackage] = Field(
        None, description="Evidence package record",
    )
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )


class GasEstimateResponse(BaseModel):
    """Response for a gas estimation operation.

    Attributes:
        success: Whether the estimation succeeded.
        cost: The gas cost estimation record.
        error: Error message (if failed).
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    cost: Optional[GasCost] = Field(
        None, description="Gas cost estimation",
    )
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )


class BatchResponse(BaseModel):
    """Response for a batch processing job submission.

    Attributes:
        success: Whether the batch was submitted.
        job: The batch job record.
        error: Error message (if failed).
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    success: bool = Field(..., description="Operation success flag")
    job: Optional[BatchJob] = Field(
        None, description="Batch job record",
    )
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(
        default=0.0, description="Processing duration in ms",
    )


class HealthResponse(BaseModel):
    """Health check response for the blockchain integration service.

    Attributes:
        status: Overall service health status (healthy/degraded/unhealthy).
        version: Service version string.
        chains: Connection status per blockchain network.
        contracts: Deployed contract count per chain.
        pending_anchors: Number of pending anchor submissions.
        active_listeners: Number of active event listeners.
        uptime_seconds: Service uptime in seconds.
        timestamp: UTC timestamp of the health check.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    status: str = Field(..., description="Service health status")
    version: str = Field(default=VERSION, description="Service version")
    chains: Dict[str, str] = Field(
        default_factory=dict,
        description="Connection status per chain",
    )
    contracts: Dict[str, int] = Field(
        default_factory=dict,
        description="Deployed contract count per chain",
    )
    pending_anchors: int = Field(
        default=0, ge=0, description="Pending anchor submissions",
    )
    active_listeners: int = Field(
        default=0, ge=0, description="Active event listeners",
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0, description="Service uptime in seconds",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of health check",
    )


class DashboardResponse(BaseModel):
    """Dashboard summary response for the blockchain integration service.

    Attributes:
        total_anchors: Total number of anchor records.
        confirmed_anchors: Number of confirmed anchors.
        pending_anchors: Number of pending anchors.
        failed_anchors: Number of failed anchors.
        total_merkle_trees: Total Merkle trees constructed.
        total_verifications: Total verification operations.
        tampered_count: Number of tampered anchors detected.
        total_gas_spent_wei: Total gas spent across all chains.
        total_access_grants: Total access grants issued.
        total_evidence_packages: Total evidence packages created.
        chains_connected: Number of connected blockchain networks.
        active_listeners: Number of active event listeners.
        timestamp: UTC timestamp of the dashboard snapshot.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    total_anchors: int = Field(
        default=0, ge=0, description="Total anchor records",
    )
    confirmed_anchors: int = Field(
        default=0, ge=0, description="Confirmed anchors",
    )
    pending_anchors: int = Field(
        default=0, ge=0, description="Pending anchors",
    )
    failed_anchors: int = Field(
        default=0, ge=0, description="Failed anchors",
    )
    total_merkle_trees: int = Field(
        default=0, ge=0, description="Total Merkle trees",
    )
    total_verifications: int = Field(
        default=0, ge=0, description="Total verifications",
    )
    tampered_count: int = Field(
        default=0, ge=0, description="Tampered anchors detected",
    )
    total_gas_spent_wei: int = Field(
        default=0, ge=0, description="Total gas spent (wei)",
    )
    total_access_grants: int = Field(
        default=0, ge=0, description="Total access grants",
    )
    total_evidence_packages: int = Field(
        default=0, ge=0, description="Total evidence packages",
    )
    chains_connected: int = Field(
        default=0, ge=0, description="Connected chains",
    )
    active_listeners: int = Field(
        default=0, ge=0, description="Active event listeners",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of dashboard snapshot",
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Constants
    "VERSION",
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    "SUPPORTED_CHAINS",
    "SUPPORTED_HASH_ALGORITHMS",
    "SUPPORTED_ANCHOR_EVENT_TYPES",
    "SUPPORTED_CONTRACT_TYPES",
    "SUPPORTED_EVIDENCE_FORMATS",
    "DEFAULT_CONFIRMATION_DEPTHS",
    # Enumerations
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
    # Core Models
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
    # Request Models
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
    # Response Models
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
]
