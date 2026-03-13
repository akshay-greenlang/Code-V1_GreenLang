# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-013 Blockchain Integration

Pydantic v2 request/response models for all Blockchain Integration REST API
endpoints. Organized by domain: anchoring, smart contracts, multi-chain
connections, verification, events, Merkle proofs, cross-party sharing,
evidence packages, batch jobs, and health.

All models use ``ConfigDict(from_attributes=True)`` for ORM compatibility
and ``Field()`` with descriptions for OpenAPI documentation.

All hash and verification fields use deterministic algorithms (SHA-256,
Keccak-256) required by EUDR Article 14 and on-chain anchoring standards.

Model Count: 80+ schemas covering 37 endpoints.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013, Section 7.4
Agent ID: GL-EUDR-BCI-013
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# =============================================================================
# Enumerations (API-layer mirrors of domain enums)
# =============================================================================


class BlockchainNetworkSchema(str, Enum):
    """Supported blockchain networks for EUDR compliance anchoring."""

    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    FABRIC = "fabric"
    BESU = "besu"


class AnchorStatusSchema(str, Enum):
    """Status of an on-chain anchor record."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    EXPIRED = "expired"


class AnchorEventTypeSchema(str, Enum):
    """Type of EUDR supply chain event being anchored on-chain."""

    DDS_SUBMISSION = "dds_submission"
    CUSTODY_TRANSFER = "custody_transfer"
    BATCH_EVENT = "batch_event"
    CERTIFICATE_REFERENCE = "certificate_reference"
    RECONCILIATION_RESULT = "reconciliation_result"
    MASS_BALANCE_ENTRY = "mass_balance_entry"
    DOCUMENT_AUTHENTICATION = "document_authentication"
    GEOLOCATION_VERIFICATION = "geolocation_verification"


class AnchorPrioritySchema(str, Enum):
    """Priority level for anchor submission scheduling."""

    P0_IMMEDIATE = "p0_immediate"
    P1_STANDARD = "p1_standard"
    P2_BATCH = "p2_batch"


class ContractTypeSchema(str, Enum):
    """Type of smart contract deployed for EUDR compliance."""

    ANCHOR_REGISTRY = "anchor_registry"
    CUSTODY_TRANSFER = "custody_transfer"
    COMPLIANCE_CHECK = "compliance_check"


class ContractStatusSchema(str, Enum):
    """Lifecycle status of a deployed smart contract."""

    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    PAUSED = "paused"
    DEPRECATED = "deprecated"


class VerificationStatusSchema(str, Enum):
    """Result status of an anchor verification operation."""

    VERIFIED = "verified"
    TAMPERED = "tampered"
    NOT_FOUND = "not_found"
    ERROR = "error"


class EventTypeSchema(str, Enum):
    """Type of on-chain event emitted by EUDR smart contracts."""

    ANCHOR_CREATED = "anchor_created"
    CUSTODY_TRANSFER_RECORDED = "custody_transfer_recorded"
    COMPLIANCE_CHECK_COMPLETED = "compliance_check_completed"
    PARTY_REGISTERED = "party_registered"


class AccessLevelSchema(str, Enum):
    """Access level for cross-party data sharing grants."""

    OPERATOR = "operator"
    COMPETENT_AUTHORITY = "competent_authority"
    AUDITOR = "auditor"
    SUPPLY_CHAIN_PARTNER = "supply_chain_partner"


class AccessStatusSchema(str, Enum):
    """Status of a cross-party data access grant."""

    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class EvidenceFormatSchema(str, Enum):
    """Output format for EUDR compliance evidence packages."""

    JSON = "json"
    PDF = "pdf"
    EUDR_XML = "eudr_xml"


class ProofFormatSchema(str, Enum):
    """Output format for Merkle proofs."""

    JSON = "json"
    BINARY = "binary"


class TransactionStatusSchema(str, Enum):
    """Status of a blockchain transaction."""

    PENDING = "pending"
    MINED = "mined"
    CONFIRMED = "confirmed"
    REVERTED = "reverted"


class ChainConnectionStatusSchema(str, Enum):
    """Status of a blockchain network connection."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class BatchJobStatusSchema(str, Enum):
    """Status of a batch processing job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobTypeSchema(str, Enum):
    """Type of batch processing job."""

    ANCHOR_BATCH = "anchor_batch"
    VERIFY_BATCH = "verify_batch"
    EVIDENCE_BATCH = "evidence_batch"
    MERKLE_BUILD = "merkle_build"
    EVENT_REPLAY = "event_replay"


class HashAlgorithmSchema(str, Enum):
    """Cryptographic hash algorithm for Merkle tree construction."""

    SHA256 = "sha256"
    SHA512 = "sha512"
    KECCAK256 = "keccak256"


class SubscriptionStatusSchema(str, Enum):
    """Status of an on-chain event subscription."""

    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    ERROR = "error"


class EvidenceStatusSchema(str, Enum):
    """Status of a compliance evidence package."""

    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
    EXPIRED = "expired"


class EvidenceVerifyStatusSchema(str, Enum):
    """Verification status of an evidence package."""

    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    TAMPERED = "tampered"


# =============================================================================
# Shared Models
# =============================================================================


class ProvenanceInfo(BaseModel):
    """Provenance tracking metadata for audit trails.

    Attributes:
        provenance_hash: SHA-256 hash of the operation data.
        algorithm: Hash algorithm used (always sha256).
        created_at: Timestamp when the provenance was recorded.
    """

    model_config = ConfigDict(from_attributes=True)

    provenance_hash: str = Field(
        ..., description="SHA-256 hash of the operation data"
    )
    algorithm: str = Field(
        default="sha256", description="Hash algorithm used"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Provenance timestamp"
    )


class PaginatedMeta(BaseModel):
    """Pagination metadata for list responses.

    Attributes:
        total: Total number of records matching the query.
        limit: Maximum records per page.
        offset: Number of records skipped.
        has_more: Whether more records exist beyond this page.
    """

    model_config = ConfigDict(from_attributes=True)

    total: int = Field(..., ge=0, description="Total matching records")
    limit: int = Field(..., ge=1, description="Records per page")
    offset: int = Field(..., ge=0, description="Records skipped")
    has_more: bool = Field(..., description="More records exist")


class TransactionInfo(BaseModel):
    """Blockchain transaction information.

    Attributes:
        tx_hash: Transaction hash on the blockchain.
        block_number: Block number containing the transaction.
        block_hash: Hash of the block containing the transaction.
        chain: Blockchain network the transaction is on.
        gas_used: Gas consumed by the transaction.
        status: Transaction status.
    """

    model_config = ConfigDict(from_attributes=True)

    tx_hash: str = Field(..., description="Transaction hash")
    block_number: Optional[int] = Field(
        None, ge=0, description="Block number"
    )
    block_hash: Optional[str] = Field(None, description="Block hash")
    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network"
    )
    gas_used: Optional[int] = Field(None, ge=0, description="Gas consumed")
    status: TransactionStatusSchema = Field(
        ..., description="Transaction status"
    )


# =============================================================================
# Anchor Schemas
# =============================================================================


class AnchorCreateRequest(BaseModel):
    """Request to create a single on-chain anchor record.

    Attributes:
        record_id: Unique identifier of the EUDR compliance record.
        event_type: Type of EUDR supply chain event.
        data_hash: SHA-256 hash of the data being anchored.
        chain: Target blockchain network.
        priority: Anchor submission priority.
        metadata: Additional metadata key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    record_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Unique record identifier",
    )
    event_type: AnchorEventTypeSchema = Field(
        ..., description="Type of EUDR event being anchored"
    )
    data_hash: str = Field(
        ..., min_length=64, max_length=128,
        description="SHA-256 hash of the data to anchor",
    )
    chain: BlockchainNetworkSchema = Field(
        default=BlockchainNetworkSchema.POLYGON,
        description="Target blockchain network",
    )
    priority: AnchorPrioritySchema = Field(
        default=AnchorPrioritySchema.P1_STANDARD,
        description="Submission priority level",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata key-value pairs"
    )

    @field_validator("data_hash")
    @classmethod
    def validate_data_hash(cls, v: str) -> str:
        """Validate data_hash is a valid hex string."""
        v = v.strip().lower()
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("data_hash must be a valid hexadecimal string")
        return v


class AnchorBatchRequest(BaseModel):
    """Request to batch anchor multiple records.

    Attributes:
        records: List of anchor create requests (max 500).
        chain: Target blockchain network for all records.
        priority: Submission priority for the batch.
        build_merkle_tree: Whether to build a Merkle tree over the batch.
    """

    model_config = ConfigDict(from_attributes=True)

    records: List[AnchorCreateRequest] = Field(
        ..., min_length=1, max_length=500,
        description="Records to anchor (max 500)",
    )
    chain: BlockchainNetworkSchema = Field(
        default=BlockchainNetworkSchema.POLYGON,
        description="Target blockchain network for all records",
    )
    priority: AnchorPrioritySchema = Field(
        default=AnchorPrioritySchema.P2_BATCH,
        description="Submission priority for the batch",
    )
    build_merkle_tree: bool = Field(
        default=True,
        description="Build Merkle tree over batch records",
    )

    @field_validator("records")
    @classmethod
    def validate_records_not_empty(cls, v: List) -> List:
        """Validate at least one record is provided."""
        if not v:
            raise ValueError("At least one record is required")
        return v


class AnchorResponse(BaseModel):
    """Response for a single anchor operation.

    Attributes:
        anchor_id: Unique anchor identifier.
        record_id: Original record identifier.
        event_type: Type of EUDR event anchored.
        data_hash: Data hash that was anchored.
        chain: Blockchain network.
        status: Current anchor status.
        tx_hash: Blockchain transaction hash (if submitted).
        block_number: Block number (if confirmed).
        priority: Submission priority.
        metadata: Additional metadata.
        created_at: Anchor creation timestamp.
        confirmed_at: Confirmation timestamp (if confirmed).
        provenance: Provenance tracking information.
    """

    model_config = ConfigDict(from_attributes=True)

    anchor_id: str = Field(
        default_factory=_new_id, description="Unique anchor identifier"
    )
    record_id: str = Field(..., description="Original record identifier")
    event_type: AnchorEventTypeSchema = Field(
        ..., description="Type of EUDR event anchored"
    )
    data_hash: str = Field(..., description="Data hash anchored")
    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network"
    )
    status: AnchorStatusSchema = Field(
        default=AnchorStatusSchema.PENDING,
        description="Current anchor status",
    )
    tx_hash: Optional[str] = Field(
        None, description="Blockchain transaction hash"
    )
    block_number: Optional[int] = Field(
        None, ge=0, description="Block number"
    )
    priority: AnchorPrioritySchema = Field(
        ..., description="Submission priority"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    confirmed_at: Optional[datetime] = Field(
        None, description="Confirmation timestamp"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )


class AnchorListResponse(BaseModel):
    """Paginated list of anchor records.

    Attributes:
        anchors: List of anchor responses.
        pagination: Pagination metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    anchors: List[AnchorResponse] = Field(
        default_factory=list, description="Anchor records"
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")


# =============================================================================
# Contract Schemas
# =============================================================================


class ContractDeployRequest(BaseModel):
    """Request to deploy a smart contract.

    Attributes:
        contract_type: Type of EUDR contract to deploy.
        chain: Target blockchain network.
        name: Human-readable contract name.
        description: Contract purpose description.
        constructor_args: Constructor arguments for deployment.
        gas_limit: Maximum gas for deployment transaction.
    """

    model_config = ConfigDict(from_attributes=True)

    contract_type: ContractTypeSchema = Field(
        ..., description="Type of EUDR contract to deploy"
    )
    chain: BlockchainNetworkSchema = Field(
        default=BlockchainNetworkSchema.POLYGON,
        description="Target blockchain network",
    )
    name: str = Field(
        ..., min_length=1, max_length=255,
        description="Human-readable contract name",
    )
    description: Optional[str] = Field(
        None, max_length=2000,
        description="Contract purpose description",
    )
    constructor_args: Optional[Dict[str, Any]] = Field(
        default=None, description="Constructor arguments"
    )
    gas_limit: Optional[int] = Field(
        None, ge=21000, le=30000000,
        description="Maximum gas for deployment",
    )


class ContractCallRequest(BaseModel):
    """Request to call a smart contract method.

    Attributes:
        method: Contract method name to call.
        args: Method arguments.
        gas_limit: Maximum gas for the call.
        value_wei: ETH/MATIC value to send (in wei).
        is_read_only: Whether this is a read-only call (no transaction).
    """

    model_config = ConfigDict(from_attributes=True)

    method: str = Field(
        ..., min_length=1, max_length=255,
        description="Contract method name",
    )
    args: Optional[Dict[str, Any]] = Field(
        default=None, description="Method arguments"
    )
    gas_limit: Optional[int] = Field(
        None, ge=21000, le=30000000,
        description="Maximum gas for the call",
    )
    value_wei: Optional[int] = Field(
        None, ge=0,
        description="Value to send in wei",
    )
    is_read_only: bool = Field(
        default=False,
        description="Read-only call (no transaction submitted)",
    )


class ContractResponse(BaseModel):
    """Response for a smart contract operation.

    Attributes:
        contract_id: Unique contract identifier.
        contract_type: Type of EUDR contract.
        chain: Blockchain network.
        address: Deployed contract address.
        status: Contract lifecycle status.
        name: Human-readable contract name.
        description: Contract purpose description.
        tx_hash: Deployment transaction hash.
        block_number: Deployment block number.
        abi_hash: SHA-256 hash of the contract ABI.
        deployed_at: Deployment timestamp.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    contract_id: str = Field(
        default_factory=_new_id, description="Unique contract identifier"
    )
    contract_type: ContractTypeSchema = Field(
        ..., description="Type of EUDR contract"
    )
    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network"
    )
    address: Optional[str] = Field(
        None, description="Deployed contract address"
    )
    status: ContractStatusSchema = Field(
        default=ContractStatusSchema.DEPLOYING,
        description="Contract lifecycle status",
    )
    name: str = Field(..., description="Contract name")
    description: Optional[str] = Field(
        None, description="Contract description"
    )
    tx_hash: Optional[str] = Field(
        None, description="Deployment transaction hash"
    )
    block_number: Optional[int] = Field(
        None, ge=0, description="Deployment block number"
    )
    abi_hash: Optional[str] = Field(
        None, description="SHA-256 hash of contract ABI"
    )
    deployed_at: Optional[datetime] = Field(
        None, description="Deployment timestamp"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )


class ContractListResponse(BaseModel):
    """Paginated list of deployed contracts.

    Attributes:
        contracts: List of contract responses.
        pagination: Pagination metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    contracts: List[ContractResponse] = Field(
        default_factory=list, description="Contract records"
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")


class ContractStateResponse(BaseModel):
    """Response for contract state query.

    Attributes:
        contract_id: Contract identifier.
        address: Contract address.
        chain: Blockchain network.
        state: Current on-chain state variables.
        block_number: Block number at which state was read.
        queried_at: Timestamp of the state query.
    """

    model_config = ConfigDict(from_attributes=True)

    contract_id: str = Field(..., description="Contract identifier")
    address: str = Field(..., description="Contract address")
    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network"
    )
    state: Dict[str, Any] = Field(
        default_factory=dict, description="On-chain state variables"
    )
    block_number: int = Field(..., ge=0, description="State block number")
    queried_at: datetime = Field(
        default_factory=_utcnow, description="Query timestamp"
    )


class ContractCallResponse(BaseModel):
    """Response for a contract method call.

    Attributes:
        contract_id: Contract identifier.
        method: Method that was called.
        result: Return value from the call.
        tx_hash: Transaction hash (if write call).
        gas_used: Gas consumed by the call.
        is_read_only: Whether this was a read-only call.
        executed_at: Execution timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    contract_id: str = Field(..., description="Contract identifier")
    method: str = Field(..., description="Method called")
    result: Optional[Any] = Field(None, description="Call return value")
    tx_hash: Optional[str] = Field(
        None, description="Transaction hash (write calls only)"
    )
    gas_used: Optional[int] = Field(
        None, ge=0, description="Gas consumed"
    )
    is_read_only: bool = Field(..., description="Whether read-only call")
    executed_at: datetime = Field(
        default_factory=_utcnow, description="Execution timestamp"
    )


# =============================================================================
# Chain Connection Schemas
# =============================================================================


class ChainConnectRequest(BaseModel):
    """Request to connect to a blockchain network.

    Attributes:
        chain: Blockchain network to connect to.
        rpc_url: RPC endpoint URL.
        ws_url: WebSocket endpoint URL (optional, for event listening).
        api_key: API key for authenticated RPC endpoints.
        confirmation_depth: Required confirmation depth.
        max_connections: Maximum concurrent connections.
    """

    model_config = ConfigDict(from_attributes=True)

    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network to connect to"
    )
    rpc_url: str = Field(
        ..., min_length=1, max_length=2048,
        description="RPC endpoint URL",
    )
    ws_url: Optional[str] = Field(
        None, max_length=2048,
        description="WebSocket endpoint URL for event listening",
    )
    api_key: Optional[str] = Field(
        None, max_length=512,
        description="API key for authenticated RPC endpoints",
    )
    confirmation_depth: Optional[int] = Field(
        None, ge=1, le=256,
        description="Required confirmation depth",
    )
    max_connections: int = Field(
        default=5, ge=1, le=50,
        description="Maximum concurrent connections",
    )

    @field_validator("rpc_url")
    @classmethod
    def validate_rpc_url(cls, v: str) -> str:
        """Validate RPC URL format."""
        v = v.strip()
        if not v.startswith(("http://", "https://", "ws://", "wss://")):
            raise ValueError(
                "rpc_url must start with http://, https://, ws://, or wss://"
            )
        return v


class GasEstimateRequest(BaseModel):
    """Request to estimate gas for a blockchain operation.

    Attributes:
        chain: Target blockchain network.
        operation: Type of operation to estimate.
        data_size_bytes: Size of the data payload in bytes.
        contract_id: Contract identifier (for contract calls).
        method: Contract method (for contract calls).
    """

    model_config = ConfigDict(from_attributes=True)

    chain: BlockchainNetworkSchema = Field(
        ..., description="Target blockchain network"
    )
    operation: str = Field(
        ..., min_length=1, max_length=255,
        description="Operation type (anchor, deploy, call)",
    )
    data_size_bytes: Optional[int] = Field(
        None, ge=0, le=100000,
        description="Data payload size in bytes",
    )
    contract_id: Optional[str] = Field(
        None, description="Contract identifier"
    )
    method: Optional[str] = Field(
        None, description="Contract method name"
    )


class ChainStatusResponse(BaseModel):
    """Response for chain connection status.

    Attributes:
        chain_id: Unique connection identifier.
        chain: Blockchain network.
        status: Connection status.
        rpc_url: RPC endpoint URL.
        block_height: Current block height.
        peer_count: Number of connected peers.
        confirmation_depth: Required confirmation depth.
        latency_ms: RPC latency in milliseconds.
        connected_at: Connection establishment timestamp.
        last_heartbeat: Last successful heartbeat timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    chain_id: str = Field(
        default_factory=_new_id, description="Unique connection identifier"
    )
    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network"
    )
    status: ChainConnectionStatusSchema = Field(
        ..., description="Connection status"
    )
    rpc_url: str = Field(..., description="RPC endpoint URL")
    block_height: Optional[int] = Field(
        None, ge=0, description="Current block height"
    )
    peer_count: Optional[int] = Field(
        None, ge=0, description="Connected peers"
    )
    confirmation_depth: int = Field(
        ..., ge=1, description="Required confirmation depth"
    )
    latency_ms: Optional[float] = Field(
        None, ge=0.0, description="RPC latency in milliseconds"
    )
    connected_at: Optional[datetime] = Field(
        None, description="Connection timestamp"
    )
    last_heartbeat: Optional[datetime] = Field(
        None, description="Last heartbeat timestamp"
    )


class ChainListResponse(BaseModel):
    """List of connected blockchain networks.

    Attributes:
        chains: List of chain status responses.
        total: Total number of connected chains.
    """

    model_config = ConfigDict(from_attributes=True)

    chains: List[ChainStatusResponse] = Field(
        default_factory=list, description="Connected chains"
    )
    total: int = Field(default=0, ge=0, description="Total connected chains")


class GasEstimateResponse(BaseModel):
    """Response for gas estimation.

    Attributes:
        chain: Blockchain network.
        operation: Operation type estimated.
        gas_estimate: Estimated gas units.
        gas_price_gwei: Current gas price in gwei.
        cost_estimate_usd: Estimated cost in USD.
        cost_estimate_native: Estimated cost in native token.
        native_token: Native token symbol (ETH, MATIC, etc.).
        estimated_at: Estimation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network"
    )
    operation: str = Field(..., description="Operation type")
    gas_estimate: int = Field(..., ge=0, description="Estimated gas units")
    gas_price_gwei: Decimal = Field(
        ..., ge=0, description="Current gas price in gwei"
    )
    cost_estimate_usd: Decimal = Field(
        ..., ge=0, description="Estimated cost in USD"
    )
    cost_estimate_native: Decimal = Field(
        ..., ge=0, description="Cost in native token"
    )
    native_token: str = Field(
        ..., description="Native token symbol"
    )
    estimated_at: datetime = Field(
        default_factory=_utcnow, description="Estimation timestamp"
    )


# =============================================================================
# Verification Schemas
# =============================================================================


class VerifyRecordRequest(BaseModel):
    """Request to verify a record against its on-chain anchor.

    Attributes:
        anchor_id: Anchor identifier to verify against.
        data_hash: Current SHA-256 hash of the data to verify.
        chain: Blockchain network to verify on.
    """

    model_config = ConfigDict(from_attributes=True)

    anchor_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Anchor identifier to verify against",
    )
    data_hash: str = Field(
        ..., min_length=64, max_length=128,
        description="Current SHA-256 hash of data to verify",
    )
    chain: Optional[BlockchainNetworkSchema] = Field(
        None, description="Blockchain network (auto-detected if omitted)"
    )

    @field_validator("data_hash")
    @classmethod
    def validate_data_hash(cls, v: str) -> str:
        """Validate data_hash is a valid hex string."""
        v = v.strip().lower()
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("data_hash must be a valid hexadecimal string")
        return v


class VerifyBatchRequest(BaseModel):
    """Request to batch verify multiple records.

    Attributes:
        records: List of verification requests (max 100).
    """

    model_config = ConfigDict(from_attributes=True)

    records: List[VerifyRecordRequest] = Field(
        ..., min_length=1, max_length=100,
        description="Records to verify (max 100)",
    )


class VerifyMerkleProofRequest(BaseModel):
    """Request to verify a Merkle inclusion proof.

    Attributes:
        leaf_hash: Hash of the leaf to verify.
        proof: List of sibling hashes forming the proof path.
        root_hash: Expected Merkle root hash.
        tree_id: Merkle tree identifier (optional).
        leaf_index: Index of the leaf in the tree.
    """

    model_config = ConfigDict(from_attributes=True)

    leaf_hash: str = Field(
        ..., min_length=64, max_length=128,
        description="Hash of the leaf to verify",
    )
    proof: List[str] = Field(
        ..., min_length=1, max_length=256,
        description="Sibling hashes forming the proof path",
    )
    root_hash: str = Field(
        ..., min_length=64, max_length=128,
        description="Expected Merkle root hash",
    )
    tree_id: Optional[str] = Field(
        None, description="Merkle tree identifier"
    )
    leaf_index: Optional[int] = Field(
        None, ge=0, description="Leaf index in the tree"
    )

    @field_validator("leaf_hash", "root_hash")
    @classmethod
    def validate_hex_hash(cls, v: str) -> str:
        """Validate hash is a valid hex string."""
        v = v.strip().lower()
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("Hash must be a valid hexadecimal string")
        return v


class VerificationResponse(BaseModel):
    """Response for a verification operation.

    Attributes:
        verification_id: Unique verification identifier.
        anchor_id: Anchor identifier verified against.
        status: Verification result status.
        data_hash: Data hash that was verified.
        on_chain_hash: Hash stored on-chain.
        chain: Blockchain network.
        tx_hash: Transaction hash of the original anchor.
        block_number: Block number of the original anchor.
        verified_at: Verification timestamp.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    verification_id: str = Field(
        default_factory=_new_id, description="Unique verification identifier"
    )
    anchor_id: str = Field(
        ..., description="Anchor identifier verified against"
    )
    status: VerificationStatusSchema = Field(
        ..., description="Verification result"
    )
    data_hash: str = Field(..., description="Data hash verified")
    on_chain_hash: Optional[str] = Field(
        None, description="Hash stored on-chain"
    )
    chain: Optional[BlockchainNetworkSchema] = Field(
        None, description="Blockchain network"
    )
    tx_hash: Optional[str] = Field(
        None, description="Original anchor transaction hash"
    )
    block_number: Optional[int] = Field(
        None, ge=0, description="Original anchor block number"
    )
    verified_at: datetime = Field(
        default_factory=_utcnow, description="Verification timestamp"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )


class VerificationListResponse(BaseModel):
    """Paginated list of verification results.

    Attributes:
        verifications: List of verification responses.
        summary: Batch verification summary.
        pagination: Pagination metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    verifications: List[VerificationResponse] = Field(
        default_factory=list, description="Verification results"
    )
    summary: Optional[Dict[str, int]] = Field(
        None, description="Summary counts by status"
    )
    pagination: Optional[PaginatedMeta] = Field(
        None, description="Pagination metadata"
    )


# =============================================================================
# Event Schemas
# =============================================================================


class EventSubscribeRequest(BaseModel):
    """Request to subscribe to on-chain events.

    Attributes:
        chain: Blockchain network to listen on.
        contract_id: Contract identifier to listen to.
        event_types: Types of events to subscribe to.
        from_block: Starting block number (default: latest).
        callback_url: Webhook URL for event notifications.
        filters: Additional event filter parameters.
    """

    model_config = ConfigDict(from_attributes=True)

    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network"
    )
    contract_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Contract identifier to listen to",
    )
    event_types: List[EventTypeSchema] = Field(
        ..., min_length=1,
        description="Types of events to subscribe to",
    )
    from_block: Optional[int] = Field(
        None, ge=0, description="Starting block number (latest if omitted)"
    )
    callback_url: Optional[str] = Field(
        None, max_length=2048,
        description="Webhook URL for notifications",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional event filter parameters"
    )


class EventReplayRequest(BaseModel):
    """Request to replay events from a specific block.

    Attributes:
        chain: Blockchain network.
        contract_id: Contract identifier.
        from_block: Starting block number.
        to_block: Ending block number (latest if omitted).
        event_types: Types of events to replay (all if omitted).
    """

    model_config = ConfigDict(from_attributes=True)

    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network"
    )
    contract_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Contract identifier",
    )
    from_block: int = Field(
        ..., ge=0, description="Starting block number"
    )
    to_block: Optional[int] = Field(
        None, ge=0, description="Ending block number (latest if omitted)"
    )
    event_types: Optional[List[EventTypeSchema]] = Field(
        default=None, description="Event types to replay (all if omitted)"
    )

    @model_validator(mode="after")
    def validate_block_range(self) -> "EventReplayRequest":
        """Validate to_block >= from_block when both are set."""
        if self.to_block is not None and self.to_block < self.from_block:
            raise ValueError("to_block must be >= from_block")
        return self


class EventQueryRequest(BaseModel):
    """Query parameters for searching indexed events.

    Attributes:
        chain: Filter by blockchain network.
        contract_id: Filter by contract identifier.
        event_type: Filter by event type.
        from_block: Start block filter.
        to_block: End block filter.
        from_date: Start date filter.
        to_date: End date filter.
    """

    model_config = ConfigDict(from_attributes=True)

    chain: Optional[BlockchainNetworkSchema] = Field(
        None, description="Filter by blockchain network"
    )
    contract_id: Optional[str] = Field(
        None, description="Filter by contract identifier"
    )
    event_type: Optional[EventTypeSchema] = Field(
        None, description="Filter by event type"
    )
    from_block: Optional[int] = Field(
        None, ge=0, description="Start block filter"
    )
    to_block: Optional[int] = Field(
        None, ge=0, description="End block filter"
    )
    from_date: Optional[datetime] = Field(
        None, description="Start date filter"
    )
    to_date: Optional[datetime] = Field(
        None, description="End date filter"
    )


class EventResponse(BaseModel):
    """Response for a single on-chain event.

    Attributes:
        event_id: Unique event identifier.
        event_type: Type of on-chain event.
        chain: Blockchain network.
        contract_id: Contract that emitted the event.
        contract_address: Contract address.
        tx_hash: Transaction hash.
        block_number: Block number.
        block_hash: Block hash.
        log_index: Log index within the transaction.
        data: Event data payload.
        indexed_at: Event indexing timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(
        default_factory=_new_id, description="Unique event identifier"
    )
    event_type: EventTypeSchema = Field(
        ..., description="Type of on-chain event"
    )
    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network"
    )
    contract_id: str = Field(
        ..., description="Contract that emitted the event"
    )
    contract_address: Optional[str] = Field(
        None, description="Contract address"
    )
    tx_hash: str = Field(..., description="Transaction hash")
    block_number: int = Field(..., ge=0, description="Block number")
    block_hash: Optional[str] = Field(None, description="Block hash")
    log_index: int = Field(..., ge=0, description="Log index in transaction")
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Event data payload"
    )
    indexed_at: datetime = Field(
        default_factory=_utcnow, description="Indexing timestamp"
    )


class EventListResponse(BaseModel):
    """Paginated list of on-chain events.

    Attributes:
        events: List of event responses.
        pagination: Pagination metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    events: List[EventResponse] = Field(
        default_factory=list, description="Event records"
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")


class SubscriptionResponse(BaseModel):
    """Response for an event subscription operation.

    Attributes:
        subscription_id: Unique subscription identifier.
        chain: Blockchain network.
        contract_id: Contract being monitored.
        event_types: Subscribed event types.
        status: Subscription status.
        from_block: Starting block number.
        callback_url: Webhook URL for notifications.
        created_at: Subscription creation timestamp.
        events_received: Count of events received so far.
    """

    model_config = ConfigDict(from_attributes=True)

    subscription_id: str = Field(
        default_factory=_new_id, description="Unique subscription identifier"
    )
    chain: BlockchainNetworkSchema = Field(
        ..., description="Blockchain network"
    )
    contract_id: str = Field(
        ..., description="Contract being monitored"
    )
    event_types: List[EventTypeSchema] = Field(
        ..., description="Subscribed event types"
    )
    status: SubscriptionStatusSchema = Field(
        default=SubscriptionStatusSchema.ACTIVE,
        description="Subscription status",
    )
    from_block: Optional[int] = Field(
        None, ge=0, description="Starting block number"
    )
    callback_url: Optional[str] = Field(
        None, description="Webhook URL"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    events_received: int = Field(
        default=0, ge=0, description="Events received count"
    )


# =============================================================================
# Merkle Proof Schemas
# =============================================================================


class MerkleBuildRequest(BaseModel):
    """Request to build a Merkle tree.

    Attributes:
        leaves: List of data hashes to include as leaves.
        algorithm: Hash algorithm for tree construction.
        sorted_tree: Whether to sort leaves before building.
        metadata: Additional metadata for the tree.
    """

    model_config = ConfigDict(from_attributes=True)

    leaves: List[str] = Field(
        ..., min_length=1, max_length=10000,
        description="Data hashes to include as leaves",
    )
    algorithm: HashAlgorithmSchema = Field(
        default=HashAlgorithmSchema.SHA256,
        description="Hash algorithm for tree construction",
    )
    sorted_tree: bool = Field(
        default=True, description="Sort leaves before building"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional tree metadata"
    )

    @field_validator("leaves")
    @classmethod
    def validate_leaves(cls, v: List[str]) -> List[str]:
        """Validate all leaves are valid hex strings."""
        validated = []
        for i, leaf in enumerate(v):
            leaf = leaf.strip().lower()
            if not leaf:
                raise ValueError(f"Leaf at index {i} is empty")
            if not all(c in "0123456789abcdef" for c in leaf):
                raise ValueError(f"Leaf at index {i} is not valid hex")
            validated.append(leaf)
        return validated


class MerkleProofRequest(BaseModel):
    """Request to generate a Merkle inclusion proof.

    Attributes:
        leaf_hash: Hash of the leaf to generate proof for.
        format: Output format for the proof.
    """

    model_config = ConfigDict(from_attributes=True)

    leaf_hash: str = Field(
        ..., min_length=64, max_length=128,
        description="Hash of the leaf",
    )
    format: ProofFormatSchema = Field(
        default=ProofFormatSchema.JSON,
        description="Output format for the proof",
    )

    @field_validator("leaf_hash")
    @classmethod
    def validate_leaf_hash(cls, v: str) -> str:
        """Validate leaf_hash is a valid hex string."""
        v = v.strip().lower()
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("leaf_hash must be a valid hexadecimal string")
        return v


class MerkleVerifyRequest(BaseModel):
    """Request to verify a standalone Merkle proof.

    Attributes:
        leaf_hash: Hash of the leaf to verify.
        proof: List of sibling hashes.
        root_hash: Expected root hash.
        algorithm: Hash algorithm used.
    """

    model_config = ConfigDict(from_attributes=True)

    leaf_hash: str = Field(
        ..., min_length=64, max_length=128,
        description="Hash of the leaf",
    )
    proof: List[str] = Field(
        ..., min_length=1, max_length=256,
        description="Sibling hashes",
    )
    root_hash: str = Field(
        ..., min_length=64, max_length=128,
        description="Expected root hash",
    )
    algorithm: HashAlgorithmSchema = Field(
        default=HashAlgorithmSchema.SHA256,
        description="Hash algorithm used",
    )

    @field_validator("leaf_hash", "root_hash")
    @classmethod
    def validate_hex_hash(cls, v: str) -> str:
        """Validate hash is valid hex."""
        v = v.strip().lower()
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("Hash must be a valid hexadecimal string")
        return v


class MerkleTreeResponse(BaseModel):
    """Response for a Merkle tree operation.

    Attributes:
        tree_id: Unique tree identifier.
        root_hash: Merkle root hash.
        algorithm: Hash algorithm used.
        leaf_count: Number of leaves in the tree.
        depth: Tree depth.
        sorted_tree: Whether leaves were sorted.
        leaves: List of leaf hashes.
        created_at: Tree creation timestamp.
        anchor_id: Associated anchor identifier (if anchored).
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    tree_id: str = Field(
        default_factory=_new_id, description="Unique tree identifier"
    )
    root_hash: str = Field(..., description="Merkle root hash")
    algorithm: HashAlgorithmSchema = Field(
        ..., description="Hash algorithm used"
    )
    leaf_count: int = Field(..., ge=1, description="Number of leaves")
    depth: int = Field(..., ge=0, description="Tree depth")
    sorted_tree: bool = Field(..., description="Whether leaves were sorted")
    leaves: Optional[List[str]] = Field(
        None, description="Leaf hashes (omitted for large trees)"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    anchor_id: Optional[str] = Field(
        None, description="Associated anchor identifier"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )


class MerkleProofResponse(BaseModel):
    """Response for a Merkle proof generation.

    Attributes:
        tree_id: Tree identifier.
        leaf_hash: Leaf hash the proof is for.
        leaf_index: Leaf position in the tree.
        proof: List of sibling hashes.
        root_hash: Merkle root hash.
        algorithm: Hash algorithm used.
        format: Proof format.
        generated_at: Generation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    tree_id: str = Field(..., description="Tree identifier")
    leaf_hash: str = Field(..., description="Leaf hash")
    leaf_index: int = Field(..., ge=0, description="Leaf index")
    proof: List[str] = Field(..., description="Sibling hashes")
    root_hash: str = Field(..., description="Merkle root hash")
    algorithm: HashAlgorithmSchema = Field(
        ..., description="Hash algorithm"
    )
    format: ProofFormatSchema = Field(
        ..., description="Proof format"
    )
    generated_at: datetime = Field(
        default_factory=_utcnow, description="Generation timestamp"
    )


class MerkleVerifyResponse(BaseModel):
    """Response for a Merkle proof verification.

    Attributes:
        is_valid: Whether the proof is valid.
        leaf_hash: Leaf hash verified.
        root_hash: Root hash verified against.
        computed_root: Root hash computed from the proof.
        algorithm: Hash algorithm used.
        verified_at: Verification timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    is_valid: bool = Field(..., description="Whether the proof is valid")
    leaf_hash: str = Field(..., description="Leaf hash verified")
    root_hash: str = Field(..., description="Expected root hash")
    computed_root: str = Field(
        ..., description="Root hash computed from proof"
    )
    algorithm: HashAlgorithmSchema = Field(
        ..., description="Hash algorithm"
    )
    verified_at: datetime = Field(
        default_factory=_utcnow, description="Verification timestamp"
    )


# =============================================================================
# Sharing / Access Grant Schemas
# =============================================================================


class AccessGrantRequest(BaseModel):
    """Request to grant cross-party data access.

    Attributes:
        record_id: Record identifier to grant access to.
        grantee_id: Identifier of the party receiving access.
        access_level: Level of access to grant.
        expires_at: Access expiry timestamp.
        scope: Specific data fields accessible (all if omitted).
        reason: Reason for granting access.
    """

    model_config = ConfigDict(from_attributes=True)

    record_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Record identifier to share",
    )
    grantee_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Party receiving access",
    )
    access_level: AccessLevelSchema = Field(
        ..., description="Level of access to grant"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Access expiry timestamp"
    )
    scope: Optional[List[str]] = Field(
        default=None,
        description="Specific data fields accessible (all if omitted)",
    )
    reason: Optional[str] = Field(
        None, max_length=2000,
        description="Reason for granting access",
    )


class AccessRevokeRequest(BaseModel):
    """Request to revoke cross-party data access.

    Attributes:
        reason: Reason for revoking access.
    """

    model_config = ConfigDict(from_attributes=True)

    reason: Optional[str] = Field(
        None, max_length=2000,
        description="Reason for revoking access",
    )


class AccessRequestRequest(BaseModel):
    """Request to request access to a record from its owner.

    Attributes:
        record_id: Record identifier to request access to.
        access_level: Requested level of access.
        justification: Justification for the access request.
        requested_duration_days: Requested access duration in days.
    """

    model_config = ConfigDict(from_attributes=True)

    record_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Record identifier to request access to",
    )
    access_level: AccessLevelSchema = Field(
        ..., description="Requested level of access"
    )
    justification: str = Field(
        ..., min_length=10, max_length=5000,
        description="Justification for the access request",
    )
    requested_duration_days: int = Field(
        default=30, ge=1, le=1825,
        description="Requested access duration in days",
    )


class MultiPartyConfirmRequest(BaseModel):
    """Request for multi-party confirmation of a data sharing agreement.

    Attributes:
        grant_id: Access grant identifier to confirm.
        confirmer_id: Identifier of the confirming party.
        confirmation_hash: SHA-256 hash of the confirmation payload.
        notes: Additional confirmation notes.
    """

    model_config = ConfigDict(from_attributes=True)

    grant_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Access grant identifier",
    )
    confirmer_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Confirming party identifier",
    )
    confirmation_hash: str = Field(
        ..., min_length=64, max_length=128,
        description="SHA-256 hash of confirmation payload",
    )
    notes: Optional[str] = Field(
        None, max_length=2000,
        description="Confirmation notes",
    )

    @field_validator("confirmation_hash")
    @classmethod
    def validate_confirmation_hash(cls, v: str) -> str:
        """Validate confirmation_hash is a valid hex string."""
        v = v.strip().lower()
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError(
                "confirmation_hash must be a valid hexadecimal string"
            )
        return v


class AccessGrantResponse(BaseModel):
    """Response for an access grant operation.

    Attributes:
        grant_id: Unique grant identifier.
        record_id: Record identifier shared.
        grantor_id: Party granting access.
        grantee_id: Party receiving access.
        access_level: Level of access granted.
        status: Grant status.
        scope: Data fields accessible.
        reason: Reason for the grant.
        granted_at: Grant timestamp.
        expires_at: Expiry timestamp.
        revoked_at: Revocation timestamp.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    grant_id: str = Field(
        default_factory=_new_id, description="Unique grant identifier"
    )
    record_id: str = Field(..., description="Record identifier")
    grantor_id: str = Field(..., description="Party granting access")
    grantee_id: str = Field(..., description="Party receiving access")
    access_level: AccessLevelSchema = Field(
        ..., description="Access level"
    )
    status: AccessStatusSchema = Field(
        default=AccessStatusSchema.ACTIVE, description="Grant status"
    )
    scope: Optional[List[str]] = Field(
        None, description="Accessible data fields"
    )
    reason: Optional[str] = Field(None, description="Grant reason")
    granted_at: datetime = Field(
        default_factory=_utcnow, description="Grant timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Expiry timestamp"
    )
    revoked_at: Optional[datetime] = Field(
        None, description="Revocation timestamp"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )


class AccessGrantListResponse(BaseModel):
    """List of access grants for a record.

    Attributes:
        grants: List of access grant responses.
        record_id: Record identifier queried.
        total: Total number of grants.
    """

    model_config = ConfigDict(from_attributes=True)

    grants: List[AccessGrantResponse] = Field(
        default_factory=list, description="Access grants"
    )
    record_id: str = Field(..., description="Record identifier")
    total: int = Field(default=0, ge=0, description="Total grants")


class SharedRecordListResponse(BaseModel):
    """List of records shared with the current party.

    Attributes:
        records: List of shared record summaries.
        total: Total shared records.
    """

    model_config = ConfigDict(from_attributes=True)

    records: List[Dict[str, Any]] = Field(
        default_factory=list, description="Shared record summaries"
    )
    total: int = Field(default=0, ge=0, description="Total shared records")


class MultiPartyConfirmResponse(BaseModel):
    """Response for a multi-party confirmation.

    Attributes:
        grant_id: Grant identifier confirmed.
        confirmer_id: Confirming party identifier.
        confirmed: Whether the confirmation was successful.
        confirmations_received: Total confirmations received.
        confirmations_required: Total confirmations required.
        fully_confirmed: Whether all required confirmations are met.
        confirmed_at: Confirmation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    grant_id: str = Field(..., description="Grant identifier")
    confirmer_id: str = Field(..., description="Confirming party")
    confirmed: bool = Field(..., description="Confirmation successful")
    confirmations_received: int = Field(
        ..., ge=0, description="Confirmations received"
    )
    confirmations_required: int = Field(
        ..., ge=1, description="Confirmations required"
    )
    fully_confirmed: bool = Field(
        ..., description="All confirmations met"
    )
    confirmed_at: datetime = Field(
        default_factory=_utcnow, description="Confirmation timestamp"
    )


# =============================================================================
# Evidence Package Schemas
# =============================================================================


class EvidencePackageRequest(BaseModel):
    """Request to generate a compliance evidence package.

    Attributes:
        record_ids: Record identifiers to include.
        format: Output format for the evidence package.
        include_merkle_proofs: Include Merkle proofs in the package.
        include_transaction_receipts: Include on-chain transaction receipts.
        include_verification_results: Include verification results.
        title: Package title.
        description: Package description.
        regulatory_framework: Regulatory framework reference.
    """

    model_config = ConfigDict(from_attributes=True)

    record_ids: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="Record identifiers to include",
    )
    format: EvidenceFormatSchema = Field(
        default=EvidenceFormatSchema.JSON,
        description="Output format",
    )
    include_merkle_proofs: bool = Field(
        default=True,
        description="Include Merkle proofs",
    )
    include_transaction_receipts: bool = Field(
        default=True,
        description="Include on-chain transaction receipts",
    )
    include_verification_results: bool = Field(
        default=True,
        description="Include verification results",
    )
    title: Optional[str] = Field(
        None, max_length=500,
        description="Package title",
    )
    description: Optional[str] = Field(
        None, max_length=5000,
        description="Package description",
    )
    regulatory_framework: str = Field(
        default="EUDR_2023_1115",
        description="Regulatory framework reference",
    )


class EvidenceVerifyRequest(BaseModel):
    """Request to verify a compliance evidence package.

    Attributes:
        package_hash: SHA-256 hash of the evidence package.
        signature: Digital signature of the package.
        package_id: Package identifier to verify (if known).
    """

    model_config = ConfigDict(from_attributes=True)

    package_hash: str = Field(
        ..., min_length=64, max_length=128,
        description="SHA-256 hash of the evidence package",
    )
    signature: Optional[str] = Field(
        None, description="Digital signature of the package"
    )
    package_id: Optional[str] = Field(
        None, description="Package identifier"
    )

    @field_validator("package_hash")
    @classmethod
    def validate_package_hash(cls, v: str) -> str:
        """Validate package_hash is a valid hex string."""
        v = v.strip().lower()
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError(
                "package_hash must be a valid hexadecimal string"
            )
        return v


class EvidencePackageResponse(BaseModel):
    """Response for an evidence package operation.

    Attributes:
        package_id: Unique package identifier.
        title: Package title.
        description: Package description.
        format: Output format.
        status: Package generation status.
        record_count: Number of records included.
        record_ids: Included record identifiers.
        package_hash: SHA-256 hash of the package.
        signature: Digital signature of the package.
        file_size_bytes: Package file size.
        regulatory_framework: Regulatory framework reference.
        created_at: Creation timestamp.
        expires_at: Package expiry timestamp.
        download_url: URL to download the package.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    package_id: str = Field(
        default_factory=_new_id, description="Unique package identifier"
    )
    title: Optional[str] = Field(None, description="Package title")
    description: Optional[str] = Field(None, description="Package description")
    format: EvidenceFormatSchema = Field(
        ..., description="Output format"
    )
    status: EvidenceStatusSchema = Field(
        default=EvidenceStatusSchema.GENERATING,
        description="Generation status",
    )
    record_count: int = Field(..., ge=0, description="Records included")
    record_ids: List[str] = Field(
        default_factory=list, description="Included record identifiers"
    )
    package_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the package"
    )
    signature: Optional[str] = Field(
        None, description="Digital signature"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="Package file size"
    )
    regulatory_framework: str = Field(
        default="EUDR_2023_1115", description="Regulatory framework"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Package expiry"
    )
    download_url: Optional[str] = Field(
        None, description="Download URL"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )


class EvidenceDownloadResponse(BaseModel):
    """Response for evidence package download.

    Attributes:
        package_id: Package identifier.
        download_url: Pre-signed download URL.
        expires_in_seconds: URL expiry in seconds.
        file_size_bytes: File size.
        content_type: MIME content type.
    """

    model_config = ConfigDict(from_attributes=True)

    package_id: str = Field(..., description="Package identifier")
    download_url: str = Field(..., description="Pre-signed download URL")
    expires_in_seconds: int = Field(
        default=3600, ge=60, description="URL expiry in seconds"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="File size"
    )
    content_type: str = Field(
        default="application/json", description="MIME content type"
    )


class EvidenceVerifyResponse(BaseModel):
    """Response for evidence package verification.

    Attributes:
        package_id: Package identifier.
        status: Verification status.
        package_hash: Hash that was verified.
        is_valid: Whether the package is valid.
        signature_valid: Whether the signature is valid.
        records_verified: Number of records verified.
        records_tampered: Number of tampered records.
        verified_at: Verification timestamp.
        details: Verification details.
    """

    model_config = ConfigDict(from_attributes=True)

    package_id: Optional[str] = Field(None, description="Package identifier")
    status: EvidenceVerifyStatusSchema = Field(
        ..., description="Verification status"
    )
    package_hash: str = Field(..., description="Hash verified")
    is_valid: bool = Field(..., description="Package valid")
    signature_valid: Optional[bool] = Field(
        None, description="Signature valid"
    )
    records_verified: int = Field(
        default=0, ge=0, description="Records verified"
    )
    records_tampered: int = Field(
        default=0, ge=0, description="Tampered records"
    )
    verified_at: datetime = Field(
        default_factory=_utcnow, description="Verification timestamp"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Verification details"
    )


# =============================================================================
# Batch Job Schemas
# =============================================================================


class SubmitBatchSchema(BaseModel):
    """Request to submit a batch processing job.

    Attributes:
        job_type: Type of batch job.
        priority: Job priority (1 = highest, 10 = lowest).
        parameters: Job-type specific parameters.
        callback_url: Webhook URL for completion notification.
    """

    model_config = ConfigDict(from_attributes=True)

    job_type: BatchJobTypeSchema = Field(
        ..., description="Type of batch job"
    )
    priority: int = Field(
        default=5, ge=1, le=10,
        description="Job priority (1=highest, 10=lowest)",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job-type specific parameters",
    )
    callback_url: Optional[str] = Field(
        None, max_length=2048,
        description="Webhook URL for completion notification",
    )


class BatchJobSchema(BaseModel):
    """Response for a batch job.

    Attributes:
        job_id: Unique job identifier.
        job_type: Type of batch job.
        status: Job status.
        priority: Job priority.
        parameters: Job parameters.
        progress_percent: Processing progress percentage.
        total_items: Total items to process.
        processed_items: Items processed so far.
        failed_items: Items that failed processing.
        result: Job result data (when completed).
        error: Error message (when failed).
        callback_url: Webhook URL.
        submitted_at: Job submission timestamp.
        started_at: Processing start timestamp.
        completed_at: Completion timestamp.
        cancelled_at: Cancellation timestamp.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(
        default_factory=_new_id, description="Unique job identifier"
    )
    job_type: BatchJobTypeSchema = Field(
        ..., description="Type of batch job"
    )
    status: BatchJobStatusSchema = Field(
        default=BatchJobStatusSchema.QUEUED, description="Job status"
    )
    priority: int = Field(
        default=5, ge=1, le=10, description="Job priority"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Job parameters"
    )
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )
    total_items: Optional[int] = Field(
        None, ge=0, description="Total items"
    )
    processed_items: Optional[int] = Field(
        None, ge=0, description="Items processed"
    )
    failed_items: Optional[int] = Field(
        None, ge=0, description="Items failed"
    )
    result: Optional[Dict[str, Any]] = Field(
        None, description="Job result"
    )
    error: Optional[str] = Field(None, description="Error message")
    callback_url: Optional[str] = Field(None, description="Webhook URL")
    submitted_at: datetime = Field(
        default_factory=_utcnow, description="Submission timestamp"
    )
    started_at: Optional[datetime] = Field(
        None, description="Start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Completion timestamp"
    )
    cancelled_at: Optional[datetime] = Field(
        None, description="Cancellation timestamp"
    )
    provenance_hash: Optional[str] = Field(
        None, description="SHA-256 hash for audit trail"
    )


class BatchJobCancelSchema(BaseModel):
    """Response for batch job cancellation.

    Attributes:
        job_id: Job identifier.
        status: Updated job status (cancelled).
        cancelled_at: Cancellation timestamp.
        message: Cancellation message.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(..., description="Job identifier")
    status: BatchJobStatusSchema = Field(
        default=BatchJobStatusSchema.CANCELLED, description="Job status"
    )
    cancelled_at: datetime = Field(
        default_factory=_utcnow, description="Cancellation timestamp"
    )
    message: str = Field(
        default="Job cancelled successfully",
        description="Cancellation message",
    )


class BatchJobResponse(BaseModel):
    """Full batch job response with status and result.

    Attributes:
        job: Batch job details.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    job: BatchJobSchema = Field(..., description="Batch job details")
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )


# =============================================================================
# Health Schema
# =============================================================================


class HealthComponentSchema(BaseModel):
    """Health status for a single service component.

    Attributes:
        name: Component name.
        status: Component health status.
        latency_ms: Component response latency.
        details: Additional health details.
    """

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., description="Component name")
    status: str = Field(default="healthy", description="Health status")
    latency_ms: Optional[float] = Field(
        None, ge=0.0, description="Response latency in ms"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Health details"
    )


class HealthSchema(BaseModel):
    """Health check response for the Blockchain Integration API.

    Attributes:
        service: Service identifier.
        status: Overall health status.
        version: Service version.
        uptime_seconds: Service uptime in seconds.
        components: Component health details.
        checked_at: Health check timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    service: str = Field(
        default="eudr-blockchain-integration",
        description="Service identifier",
    )
    status: str = Field(default="healthy", description="Overall health")
    version: str = Field(default="1.0.0", description="Service version")
    uptime_seconds: float = Field(
        default=0.0, ge=0.0, description="Uptime in seconds"
    )
    components: List[HealthComponentSchema] = Field(
        default_factory=lambda: [
            HealthComponentSchema(name="api", status="healthy"),
            HealthComponentSchema(name="database", status="healthy"),
            HealthComponentSchema(name="cache", status="healthy"),
            HealthComponentSchema(name="blockchain_connector", status="healthy"),
            HealthComponentSchema(name="event_listener", status="healthy"),
            HealthComponentSchema(name="merkle_engine", status="healthy"),
        ],
        description="Component health details",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow, description="Health check timestamp"
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "AccessLevelSchema",
    "AccessStatusSchema",
    "AnchorEventTypeSchema",
    "AnchorPrioritySchema",
    "AnchorStatusSchema",
    "BatchJobStatusSchema",
    "BatchJobTypeSchema",
    "BlockchainNetworkSchema",
    "ChainConnectionStatusSchema",
    "ContractStatusSchema",
    "ContractTypeSchema",
    "EventTypeSchema",
    "EvidenceFormatSchema",
    "EvidenceStatusSchema",
    "EvidenceVerifyStatusSchema",
    "HashAlgorithmSchema",
    "ProofFormatSchema",
    "SubscriptionStatusSchema",
    "TransactionStatusSchema",
    "VerificationStatusSchema",
    # Shared models
    "HealthComponentSchema",
    "PaginatedMeta",
    "ProvenanceInfo",
    "TransactionInfo",
    # Anchor schemas
    "AnchorBatchRequest",
    "AnchorCreateRequest",
    "AnchorListResponse",
    "AnchorResponse",
    # Contract schemas
    "ContractCallRequest",
    "ContractCallResponse",
    "ContractDeployRequest",
    "ContractListResponse",
    "ContractResponse",
    "ContractStateResponse",
    # Chain schemas
    "ChainConnectRequest",
    "ChainListResponse",
    "ChainStatusResponse",
    "GasEstimateRequest",
    "GasEstimateResponse",
    # Verification schemas
    "VerificationListResponse",
    "VerificationResponse",
    "VerifyBatchRequest",
    "VerifyMerkleProofRequest",
    "VerifyRecordRequest",
    # Event schemas
    "EventListResponse",
    "EventQueryRequest",
    "EventReplayRequest",
    "EventResponse",
    "EventSubscribeRequest",
    "SubscriptionResponse",
    # Merkle schemas
    "MerkleBuildRequest",
    "MerkleProofRequest",
    "MerkleProofResponse",
    "MerkleTreeResponse",
    "MerkleVerifyRequest",
    "MerkleVerifyResponse",
    # Sharing schemas
    "AccessGrantListResponse",
    "AccessGrantRequest",
    "AccessGrantResponse",
    "AccessRequestRequest",
    "AccessRevokeRequest",
    "MultiPartyConfirmRequest",
    "MultiPartyConfirmResponse",
    "SharedRecordListResponse",
    # Evidence schemas
    "EvidenceDownloadResponse",
    "EvidencePackageRequest",
    "EvidencePackageResponse",
    "EvidenceVerifyRequest",
    "EvidenceVerifyResponse",
    # Batch job schemas
    "BatchJobCancelSchema",
    "BatchJobResponse",
    "BatchJobSchema",
    "SubmitBatchSchema",
    # Health
    "HealthSchema",
]
