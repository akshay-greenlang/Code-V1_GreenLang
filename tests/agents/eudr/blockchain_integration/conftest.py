# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-013 Blockchain Integration Agent test suite.

Provides reusable fixtures for anchor records, Merkle trees, Merkle proofs,
smart contracts, contract events, chain connections, verification results,
access grants, evidence packages, gas costs, helper factories, assertion
helpers, reference data constants, and engine fixtures.

Sample Anchor Records:
    ANCHOR_DDS_POLYGON, ANCHOR_CUSTODY_ETHEREUM, ANCHOR_BATCH_FABRIC

Sample Merkle Trees:
    MERKLE_TREE_4_LEAVES, MERKLE_TREE_SINGLE

Sample Smart Contracts:
    CONTRACT_ANCHOR_REGISTRY_POLYGON, CONTRACT_CUSTODY_ETHEREUM

Sample Verification Results:
    VERIFICATION_VERIFIED, VERIFICATION_TAMPERED

Sample Access Grants:
    GRANT_AUTHORITY_ACTIVE, GRANT_AUDITOR_EXPIRED

Sample Evidence Packages:
    EVIDENCE_JSON_DDS, EVIDENCE_PDF_DDS

Helper Factories: make_anchor_record(), make_merkle_tree(), make_merkle_proof(),
    make_smart_contract(), make_contract_event(), make_chain_connection(),
    make_verification_result(), make_access_grant(), make_evidence_package(),
    make_gas_cost()

Assertion Helpers: assert_anchor_valid(), assert_merkle_tree_valid(),
    assert_merkle_proof_valid(), assert_contract_valid(),
    assert_verification_valid(), assert_valid_sha256(), assert_valid_tx_hash()

Reference Data Constants: BLOCKCHAIN_NETWORKS, ANCHOR_EVENT_TYPES,
    ANCHOR_STATUSES, ANCHOR_PRIORITIES, CONTRACT_TYPES, CONTRACT_STATUSES,
    VERIFICATION_STATUSES, EVENT_TYPES, ACCESS_LEVELS, ACCESS_STATUSES,
    EVIDENCE_FORMATS, PROOF_FORMATS, TRANSACTION_STATUSES,
    CONNECTION_STATUSES, BATCH_JOB_STATUSES, SHA256_HEX_LENGTH,
    CONFIRMATION_DEPTHS, EUDR_COMMODITIES

Engine Fixtures (9 engines with pytest.skip for unimplemented)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013)
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH: int = 64
SHA512_HEX_LENGTH: int = 128

BLOCKCHAIN_NETWORKS: List[str] = ["ethereum", "polygon", "fabric", "besu"]

ANCHOR_EVENT_TYPES: List[str] = [
    "dds_submission",
    "custody_transfer",
    "batch_event",
    "certificate_reference",
    "reconciliation_result",
    "mass_balance_entry",
    "document_authentication",
    "geolocation_verification",
]

ANCHOR_STATUSES: List[str] = [
    "pending", "submitted", "confirmed", "failed", "expired",
]

ANCHOR_PRIORITIES: List[str] = [
    "p0_immediate", "p1_standard", "p2_batch",
]

CONTRACT_TYPES: List[str] = [
    "anchor_registry", "custody_transfer", "compliance_check",
]

CONTRACT_STATUSES: List[str] = [
    "deploying", "deployed", "paused", "deprecated",
]

VERIFICATION_STATUSES: List[str] = [
    "verified", "tampered", "not_found", "error",
]

EVENT_TYPES: List[str] = [
    "anchor_created",
    "custody_transfer_recorded",
    "compliance_check_completed",
    "party_registered",
]

ACCESS_LEVELS: List[str] = [
    "operator", "competent_authority", "auditor", "supply_chain_partner",
]

ACCESS_STATUSES: List[str] = [
    "active", "revoked", "expired",
]

EVIDENCE_FORMATS: List[str] = ["json", "pdf", "eudr_xml"]

PROOF_FORMATS: List[str] = ["json", "binary"]

TRANSACTION_STATUSES: List[str] = [
    "pending", "mined", "confirmed", "reverted",
]

CONNECTION_STATUSES: List[str] = [
    "connected", "disconnected", "error", "reconnecting",
]

BATCH_JOB_STATUSES: List[str] = [
    "queued", "processing", "completed", "failed", "cancelled",
]

CONFIRMATION_DEPTHS: Dict[str, int] = {
    "ethereum": 12,
    "polygon": 32,
    "fabric": 1,
    "besu": 1,
}

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
]

HASH_ALGORITHMS: List[str] = ["sha256", "sha512", "keccak256"]

GAS_OPERATIONS: List[str] = ["anchor", "deploy", "verify", "batch_anchor"]

MAX_BATCH_SIZE: int = 500

EUDR_RETENTION_YEARS: int = 5

# Chain IDs for EVM networks
CHAIN_IDS: Dict[str, int] = {
    "ethereum": 1,
    "polygon": 137,
    "besu": 1337,
}


# ---------------------------------------------------------------------------
# Pre-generated Identifiers
# ---------------------------------------------------------------------------

# Operator IDs
OPERATOR_ID_EU_001 = "OP-EU-COCOA-001"
OPERATOR_ID_EU_002 = "OP-EU-WOOD-002"
OPERATOR_ID_EU_003 = "OP-EU-PALM-003"

# Anchor IDs
ANCHOR_ID_001 = "ANC-DDS-001"
ANCHOR_ID_002 = "ANC-CUST-002"
ANCHOR_ID_003 = "ANC-BATCH-003"
ANCHOR_ID_004 = "ANC-CERT-004"

# Contract IDs
CONTRACT_ID_ANCHOR_001 = "CTR-ANCHOR-001"
CONTRACT_ID_CUSTODY_001 = "CTR-CUSTODY-001"
CONTRACT_ID_COMPLIANCE_001 = "CTR-COMPLIANCE-001"

# DDS IDs
DDS_ID_001 = "DDS-2026-EU-001"
DDS_ID_002 = "DDS-2026-EU-002"

# Tree IDs
TREE_ID_001 = "TREE-001"
TREE_ID_002 = "TREE-002"

# Grant IDs
GRANT_ID_001 = "GRANT-AUTH-001"
GRANT_ID_002 = "GRANT-AUDIT-001"

# Package IDs
PACKAGE_ID_001 = "PKG-JSON-001"
PACKAGE_ID_002 = "PKG-PDF-001"

# Sample transaction hashes (Ethereum-style 0x-prefixed 64 hex chars)
SAMPLE_TX_HASH = "0x" + "a1b2c3d4e5f6" * 10 + "a1b2c3d4"
SAMPLE_TX_HASH_2 = "0x" + "f1e2d3c4b5a6" * 10 + "f1e2d3c4"

# Sample block data
SAMPLE_BLOCK_NUMBER = 18_500_000
SAMPLE_BLOCK_HASH = "0x" + "1234567890abcdef" * 4

# Sample contract addresses (Ethereum-style 0x + 40 hex chars)
SAMPLE_CONTRACT_ADDRESS = "0x" + "abcdef1234567890" * 2 + "abcdef12"
SAMPLE_CONTRACT_ADDRESS_2 = "0x" + "fedcba0987654321" * 2 + "fedcba09"
SAMPLE_DEPLOYER_ADDRESS = "0x" + "1111222233334444" * 2 + "11112222"

# Sample RPC endpoints
SAMPLE_RPC_ETHEREUM = "https://mainnet.infura.io/v3/test-project-id"
SAMPLE_RPC_POLYGON = "https://polygon-rpc.com"
SAMPLE_RPC_FABRIC = "grpcs://fabric-orderer.example.com:7050"
SAMPLE_RPC_BESU = "https://besu-node.example.com:8545"

RPC_ENDPOINTS: Dict[str, str] = {
    "ethereum": SAMPLE_RPC_ETHEREUM,
    "polygon": SAMPLE_RPC_POLYGON,
    "fabric": SAMPLE_RPC_FABRIC,
    "besu": SAMPLE_RPC_BESU,
}


# ---------------------------------------------------------------------------
# Timestamp and Hash Helpers
# ---------------------------------------------------------------------------

def _ts(days_ago: int = 0, hours_ago: int = 0) -> str:
    """Generate ISO timestamp relative to now."""
    return (
        datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    ).isoformat()


def _ts_dt(days_ago: int = 0, hours_ago: int = 0) -> datetime:
    """Generate datetime object relative to now."""
    return datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)


def _sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sha512(data: str) -> str:
    """Compute SHA-512 hex digest of a string."""
    return hashlib.sha512(data.encode("utf-8")).hexdigest()


def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash of data for provenance verification.

    Args:
        data: Data to hash (will be JSON-serialized).

    Returns:
        64-character hex digest string.
    """
    if isinstance(data, str):
        payload = data
    elif isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    else:
        payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _merkle_hash_pair(left: str, right: str) -> str:
    """Compute Merkle parent hash from two children (sorted for determinism)."""
    combined = "".join(sorted([left, right]))
    return _sha256(combined)


def _build_merkle_root(leaf_hashes: List[str]) -> str:
    """Compute Merkle root from a list of leaf hashes."""
    if not leaf_hashes:
        raise ValueError("Cannot compute Merkle root of empty list")
    if len(leaf_hashes) == 1:
        return leaf_hashes[0]
    sorted_hashes = sorted(leaf_hashes)
    level = sorted_hashes[:]
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                next_level.append(_merkle_hash_pair(level[i], level[i + 1]))
            else:
                next_level.append(level[i])
        level = next_level
    return level[0]


# Sample data hashes for anchoring
SAMPLE_DATA_HASH = _sha256("eudr-dds-submission-2026-001")
SAMPLE_DATA_HASH_2 = _sha256("eudr-custody-transfer-2026-002")
SAMPLE_DATA_HASH_3 = _sha256("eudr-batch-event-2026-003")
SAMPLE_DATA_HASH_4 = _sha256("eudr-certificate-ref-2026-004")

# Pre-computed Merkle root for the 4 sample hashes
SAMPLE_MERKLE_ROOT = _build_merkle_root([
    SAMPLE_DATA_HASH, SAMPLE_DATA_HASH_2,
    SAMPLE_DATA_HASH_3, SAMPLE_DATA_HASH_4,
])

# ABI hash for contracts
SAMPLE_ABI_HASH = _sha256("anchor-registry-abi-v1.0.0")
SAMPLE_ABI_HASH_2 = _sha256("custody-transfer-abi-v1.0.0")


# ---------------------------------------------------------------------------
# Sample Anchor Records
# ---------------------------------------------------------------------------

ANCHOR_DDS_POLYGON: Dict[str, Any] = {
    "anchor_id": ANCHOR_ID_001,
    "data_hash": SAMPLE_DATA_HASH,
    "event_type": "dds_submission",
    "chain": "polygon",
    "status": "confirmed",
    "priority": "p0_immediate",
    "merkle_root": SAMPLE_MERKLE_ROOT,
    "merkle_leaf_index": 0,
    "tx_hash": SAMPLE_TX_HASH,
    "block_number": SAMPLE_BLOCK_NUMBER,
    "block_hash": SAMPLE_BLOCK_HASH,
    "confirmations": 32,
    "required_confirmations": 32,
    "gas_used": 85_000,
    "gas_price_wei": 30_000_000_000,
    "operator_id": OPERATOR_ID_EU_001,
    "commodity": "cocoa",
    "source_agent_id": "AGENT-EUDR-004",
    "source_record_id": DDS_ID_001,
    "payload_metadata": {"dds_version": "1.0", "article": "4"},
    "retry_count": 0,
    "error_message": None,
    "created_at": _ts(days_ago=10),
    "submitted_at": _ts(days_ago=10),
    "confirmed_at": _ts(days_ago=10),
    "expires_at": _ts(days_ago=-(365 * 5)),
    "provenance_hash": None,
}

ANCHOR_CUSTODY_ETHEREUM: Dict[str, Any] = {
    "anchor_id": ANCHOR_ID_002,
    "data_hash": SAMPLE_DATA_HASH_2,
    "event_type": "custody_transfer",
    "chain": "ethereum",
    "status": "submitted",
    "priority": "p1_standard",
    "merkle_root": None,
    "merkle_leaf_index": None,
    "tx_hash": SAMPLE_TX_HASH_2,
    "block_number": None,
    "block_hash": None,
    "confirmations": 0,
    "required_confirmations": 12,
    "gas_used": None,
    "gas_price_wei": 50_000_000_000,
    "operator_id": OPERATOR_ID_EU_002,
    "commodity": "wood",
    "source_agent_id": "AGENT-EUDR-003",
    "source_record_id": "CT-2026-001",
    "payload_metadata": {"sender": "SUP-BR-001", "receiver": "OP-EU-WOOD-002"},
    "retry_count": 0,
    "error_message": None,
    "created_at": _ts(days_ago=5),
    "submitted_at": _ts(days_ago=5),
    "confirmed_at": None,
    "expires_at": _ts(days_ago=-(365 * 5)),
    "provenance_hash": None,
}

ANCHOR_BATCH_FABRIC: Dict[str, Any] = {
    "anchor_id": ANCHOR_ID_003,
    "data_hash": SAMPLE_DATA_HASH_3,
    "event_type": "batch_event",
    "chain": "fabric",
    "status": "pending",
    "priority": "p2_batch",
    "merkle_root": None,
    "merkle_leaf_index": None,
    "tx_hash": None,
    "block_number": None,
    "block_hash": None,
    "confirmations": 0,
    "required_confirmations": 1,
    "gas_used": None,
    "gas_price_wei": None,
    "operator_id": OPERATOR_ID_EU_003,
    "commodity": "oil_palm",
    "source_agent_id": "AGENT-EUDR-011",
    "source_record_id": "BATCH-2026-001",
    "payload_metadata": {"batch_type": "split"},
    "retry_count": 0,
    "error_message": None,
    "created_at": _ts(hours_ago=2),
    "submitted_at": None,
    "confirmed_at": None,
    "expires_at": _ts(days_ago=-(365 * 5)),
    "provenance_hash": None,
}

ALL_SAMPLE_ANCHORS: List[Dict[str, Any]] = [
    ANCHOR_DDS_POLYGON, ANCHOR_CUSTODY_ETHEREUM, ANCHOR_BATCH_FABRIC,
]


# ---------------------------------------------------------------------------
# Sample Merkle Trees
# ---------------------------------------------------------------------------

_TREE_4_LEAVES_HASHES = [
    SAMPLE_DATA_HASH, SAMPLE_DATA_HASH_2,
    SAMPLE_DATA_HASH_3, SAMPLE_DATA_HASH_4,
]

MERKLE_TREE_4_LEAVES: Dict[str, Any] = {
    "tree_id": TREE_ID_001,
    "root_hash": SAMPLE_MERKLE_ROOT,
    "leaf_count": 4,
    "leaves": [
        {
            "leaf_index": i,
            "data_hash": h,
            "anchor_id": f"ANC-{i:03d}",
            "leaf_hash": _sha256(f"leaf-{h}"),
        }
        for i, h in enumerate(sorted(_TREE_4_LEAVES_HASHES))
    ],
    "depth": 2,
    "hash_algorithm": "sha256",
    "sorted": True,
    "chain": "polygon",
    "tx_hash": SAMPLE_TX_HASH,
    "block_number": SAMPLE_BLOCK_NUMBER,
    "anchor_ids": [ANCHOR_ID_001, ANCHOR_ID_002, ANCHOR_ID_003, ANCHOR_ID_004],
    "created_at": _ts(days_ago=10),
    "provenance_hash": None,
}

_SINGLE_HASH = _sha256("single-leaf-data")

MERKLE_TREE_SINGLE: Dict[str, Any] = {
    "tree_id": TREE_ID_002,
    "root_hash": _SINGLE_HASH,
    "leaf_count": 1,
    "leaves": [
        {
            "leaf_index": 0,
            "data_hash": _SINGLE_HASH,
            "anchor_id": "ANC-SINGLE",
            "leaf_hash": _sha256(f"leaf-{_SINGLE_HASH}"),
        },
    ],
    "depth": 0,
    "hash_algorithm": "sha256",
    "sorted": True,
    "chain": "polygon",
    "tx_hash": None,
    "block_number": None,
    "anchor_ids": ["ANC-SINGLE"],
    "created_at": _ts(days_ago=1),
    "provenance_hash": None,
}

ALL_SAMPLE_MERKLE_TREES: List[Dict[str, Any]] = [
    MERKLE_TREE_4_LEAVES, MERKLE_TREE_SINGLE,
]


# ---------------------------------------------------------------------------
# Sample Merkle Proofs
# ---------------------------------------------------------------------------

MERKLE_PROOF_LEAF_0: Dict[str, Any] = {
    "proof_id": f"PROOF-{ANCHOR_ID_001}",
    "tree_id": TREE_ID_001,
    "root_hash": SAMPLE_MERKLE_ROOT,
    "leaf_hash": sorted(_TREE_4_LEAVES_HASHES)[0],
    "leaf_index": 0,
    "sibling_hashes": [
        sorted(_TREE_4_LEAVES_HASHES)[1],
        _merkle_hash_pair(
            sorted(_TREE_4_LEAVES_HASHES)[2],
            sorted(_TREE_4_LEAVES_HASHES)[3],
        ),
    ],
    "path_indices": [0, 0],
    "hash_algorithm": "sha256",
    "proof_format": "json",
    "anchor_id": ANCHOR_ID_001,
    "verified": True,
    "created_at": _ts(days_ago=10),
}

ALL_SAMPLE_PROOFS: List[Dict[str, Any]] = [MERKLE_PROOF_LEAF_0]


# ---------------------------------------------------------------------------
# Sample Smart Contracts
# ---------------------------------------------------------------------------

CONTRACT_ANCHOR_REGISTRY_POLYGON: Dict[str, Any] = {
    "contract_id": CONTRACT_ID_ANCHOR_001,
    "contract_type": "anchor_registry",
    "chain": "polygon",
    "address": SAMPLE_CONTRACT_ADDRESS,
    "deployer_address": SAMPLE_DEPLOYER_ADDRESS,
    "deploy_tx_hash": SAMPLE_TX_HASH,
    "deploy_block_number": SAMPLE_BLOCK_NUMBER - 1_000_000,
    "abi_hash": SAMPLE_ABI_HASH,
    "version": "1.0.0",
    "status": "deployed",
    "created_at": _ts(days_ago=90),
    "deployed_at": _ts(days_ago=90),
    "paused_at": None,
    "deprecated_at": None,
    "provenance_hash": None,
}

CONTRACT_CUSTODY_ETHEREUM: Dict[str, Any] = {
    "contract_id": CONTRACT_ID_CUSTODY_001,
    "contract_type": "custody_transfer",
    "chain": "ethereum",
    "address": SAMPLE_CONTRACT_ADDRESS_2,
    "deployer_address": SAMPLE_DEPLOYER_ADDRESS,
    "deploy_tx_hash": SAMPLE_TX_HASH_2,
    "deploy_block_number": SAMPLE_BLOCK_NUMBER - 500_000,
    "abi_hash": SAMPLE_ABI_HASH_2,
    "version": "1.0.0",
    "status": "deployed",
    "created_at": _ts(days_ago=60),
    "deployed_at": _ts(days_ago=60),
    "paused_at": None,
    "deprecated_at": None,
    "provenance_hash": None,
}

ALL_SAMPLE_CONTRACTS: List[Dict[str, Any]] = [
    CONTRACT_ANCHOR_REGISTRY_POLYGON, CONTRACT_CUSTODY_ETHEREUM,
]


# ---------------------------------------------------------------------------
# Sample Contract Events
# ---------------------------------------------------------------------------

EVENT_ANCHOR_CREATED: Dict[str, Any] = {
    "event_id": "EVT-ANC-001",
    "event_type": "anchor_created",
    "contract_address": SAMPLE_CONTRACT_ADDRESS,
    "chain": "polygon",
    "tx_hash": SAMPLE_TX_HASH,
    "block_number": SAMPLE_BLOCK_NUMBER,
    "block_hash": SAMPLE_BLOCK_HASH,
    "log_index": 0,
    "event_data": {
        "merkle_root": SAMPLE_MERKLE_ROOT,
        "leaf_count": 4,
        "operator_id": OPERATOR_ID_EU_001,
    },
    "indexed_at": _ts(days_ago=10),
    "provenance_hash": None,
}

EVENT_CUSTODY_RECORDED: Dict[str, Any] = {
    "event_id": "EVT-CUST-001",
    "event_type": "custody_transfer_recorded",
    "contract_address": SAMPLE_CONTRACT_ADDRESS_2,
    "chain": "ethereum",
    "tx_hash": SAMPLE_TX_HASH_2,
    "block_number": SAMPLE_BLOCK_NUMBER + 100,
    "block_hash": SAMPLE_BLOCK_HASH,
    "log_index": 1,
    "event_data": {
        "sender": "SUP-BR-001",
        "receiver": OPERATOR_ID_EU_002,
        "commodity": "wood",
        "quantity_kg": 25000.0,
    },
    "indexed_at": _ts(days_ago=5),
    "provenance_hash": None,
}

ALL_SAMPLE_EVENTS: List[Dict[str, Any]] = [
    EVENT_ANCHOR_CREATED, EVENT_CUSTODY_RECORDED,
]


# ---------------------------------------------------------------------------
# Sample Chain Connections
# ---------------------------------------------------------------------------

CONNECTION_POLYGON: Dict[str, Any] = {
    "connection_id": "CONN-POLYGON-001",
    "chain": "polygon",
    "rpc_endpoint": SAMPLE_RPC_POLYGON,
    "status": "connected",
    "latest_block": SAMPLE_BLOCK_NUMBER + 50_000,
    "peer_count": 25,
    "chain_id": 137,
    "network_name": "Polygon Mainnet",
    "connected_at": _ts(hours_ago=6),
    "last_heartbeat": _ts(),
    "error_message": None,
}

CONNECTION_ETHEREUM: Dict[str, Any] = {
    "connection_id": "CONN-ETH-001",
    "chain": "ethereum",
    "rpc_endpoint": SAMPLE_RPC_ETHEREUM,
    "status": "connected",
    "latest_block": SAMPLE_BLOCK_NUMBER,
    "peer_count": 50,
    "chain_id": 1,
    "network_name": "Ethereum Mainnet",
    "connected_at": _ts(hours_ago=12),
    "last_heartbeat": _ts(),
    "error_message": None,
}

CONNECTION_FABRIC: Dict[str, Any] = {
    "connection_id": "CONN-FABRIC-001",
    "chain": "fabric",
    "rpc_endpoint": SAMPLE_RPC_FABRIC,
    "status": "connected",
    "latest_block": 1_200_000,
    "peer_count": 4,
    "chain_id": None,
    "network_name": "EUDR Supply Chain Channel",
    "connected_at": _ts(hours_ago=24),
    "last_heartbeat": _ts(),
    "error_message": None,
}

CONNECTION_BESU: Dict[str, Any] = {
    "connection_id": "CONN-BESU-001",
    "chain": "besu",
    "rpc_endpoint": SAMPLE_RPC_BESU,
    "status": "disconnected",
    "latest_block": None,
    "peer_count": None,
    "chain_id": 1337,
    "network_name": "EUDR Consortium Besu",
    "connected_at": None,
    "last_heartbeat": None,
    "error_message": "Connection refused: ECONNREFUSED",
}

ALL_SAMPLE_CONNECTIONS: List[Dict[str, Any]] = [
    CONNECTION_POLYGON, CONNECTION_ETHEREUM, CONNECTION_FABRIC, CONNECTION_BESU,
]


# ---------------------------------------------------------------------------
# Sample Verification Results
# ---------------------------------------------------------------------------

VERIFICATION_VERIFIED: Dict[str, Any] = {
    "verification_id": "VER-001",
    "anchor_id": ANCHOR_ID_001,
    "status": "verified",
    "chain": "polygon",
    "merkle_proof": None,
    "on_chain_root": SAMPLE_MERKLE_ROOT,
    "computed_root": SAMPLE_MERKLE_ROOT,
    "data_hash_match": True,
    "root_hash_match": True,
    "block_number": SAMPLE_BLOCK_NUMBER,
    "gas_used": 45_000,
    "cached": False,
    "error_message": None,
    "verified_at": _ts(),
    "provenance_hash": None,
}

VERIFICATION_TAMPERED: Dict[str, Any] = {
    "verification_id": "VER-002",
    "anchor_id": ANCHOR_ID_002,
    "status": "tampered",
    "chain": "ethereum",
    "merkle_proof": None,
    "on_chain_root": SAMPLE_MERKLE_ROOT,
    "computed_root": _sha256("tampered-data-root"),
    "data_hash_match": True,
    "root_hash_match": False,
    "block_number": SAMPLE_BLOCK_NUMBER + 100,
    "gas_used": 45_000,
    "cached": False,
    "error_message": "Root hash mismatch: data has been tampered with",
    "verified_at": _ts(),
    "provenance_hash": None,
}

VERIFICATION_NOT_FOUND: Dict[str, Any] = {
    "verification_id": "VER-003",
    "anchor_id": "ANC-NONEXISTENT",
    "status": "not_found",
    "chain": "polygon",
    "merkle_proof": None,
    "on_chain_root": None,
    "computed_root": None,
    "data_hash_match": None,
    "root_hash_match": None,
    "block_number": None,
    "gas_used": None,
    "cached": False,
    "error_message": "Anchor record not found on-chain",
    "verified_at": _ts(),
    "provenance_hash": None,
}

ALL_SAMPLE_VERIFICATIONS: List[Dict[str, Any]] = [
    VERIFICATION_VERIFIED, VERIFICATION_TAMPERED, VERIFICATION_NOT_FOUND,
]


# ---------------------------------------------------------------------------
# Sample Access Grants
# ---------------------------------------------------------------------------

GRANT_AUTHORITY_ACTIVE: Dict[str, Any] = {
    "grant_id": GRANT_ID_001,
    "anchor_id": ANCHOR_ID_001,
    "grantor_id": OPERATOR_ID_EU_001,
    "grantee_id": "CA-DE-BMEL-001",
    "access_level": "competent_authority",
    "status": "active",
    "scope": {"fields": ["data_hash", "event_type", "commodity", "merkle_root"]},
    "multi_party_confirmations": 2,
    "required_confirmations": 2,
    "granted_at": _ts(days_ago=5),
    "expires_at": _ts(days_ago=-360),
    "revoked_at": None,
    "revocation_reason": None,
    "provenance_hash": None,
}

GRANT_AUDITOR_EXPIRED: Dict[str, Any] = {
    "grant_id": GRANT_ID_002,
    "anchor_id": ANCHOR_ID_002,
    "grantor_id": OPERATOR_ID_EU_002,
    "grantee_id": "AUD-KPMG-001",
    "access_level": "auditor",
    "status": "expired",
    "scope": {"audit_engagement": "AE-2025-001"},
    "multi_party_confirmations": 1,
    "required_confirmations": 2,
    "granted_at": _ts(days_ago=100),
    "expires_at": _ts(days_ago=10),
    "revoked_at": None,
    "revocation_reason": None,
    "provenance_hash": None,
}

ALL_SAMPLE_GRANTS: List[Dict[str, Any]] = [
    GRANT_AUTHORITY_ACTIVE, GRANT_AUDITOR_EXPIRED,
]


# ---------------------------------------------------------------------------
# Sample Evidence Packages
# ---------------------------------------------------------------------------

EVIDENCE_JSON_DDS: Dict[str, Any] = {
    "package_id": PACKAGE_ID_001,
    "anchor_ids": [ANCHOR_ID_001],
    "format": "json",
    "operator_id": OPERATOR_ID_EU_001,
    "merkle_proofs": [],
    "verification_results": [],
    "chain_references": {
        "tx_hash": SAMPLE_TX_HASH,
        "block_number": SAMPLE_BLOCK_NUMBER,
        "chain": "polygon",
    },
    "package_hash": _sha256(f"evidence-{PACKAGE_ID_001}"),
    "signed": False,
    "signature": None,
    "signer_id": None,
    "retention_until": _ts(days_ago=-(365 * 5)),
    "created_at": _ts(days_ago=1),
    "provenance_hash": None,
}

EVIDENCE_PDF_DDS: Dict[str, Any] = {
    "package_id": PACKAGE_ID_002,
    "anchor_ids": [ANCHOR_ID_001, ANCHOR_ID_002],
    "format": "pdf",
    "operator_id": OPERATOR_ID_EU_001,
    "merkle_proofs": [],
    "verification_results": [],
    "chain_references": {
        "tx_hashes": [SAMPLE_TX_HASH, SAMPLE_TX_HASH_2],
        "chains": ["polygon", "ethereum"],
    },
    "package_hash": _sha256(f"evidence-{PACKAGE_ID_002}"),
    "signed": True,
    "signature": "base64-encoded-pkcs7-signature-placeholder",
    "signer_id": "KEY-EUDR-SIGN-001",
    "retention_until": _ts(days_ago=-(365 * 5)),
    "created_at": _ts(days_ago=1),
    "provenance_hash": None,
}

ALL_SAMPLE_EVIDENCE: List[Dict[str, Any]] = [
    EVIDENCE_JSON_DDS, EVIDENCE_PDF_DDS,
]


# ---------------------------------------------------------------------------
# Sample Gas Costs
# ---------------------------------------------------------------------------

GAS_COST_ANCHOR_POLYGON: Dict[str, Any] = {
    "cost_id": "GAS-ANC-001",
    "chain": "polygon",
    "operation": "anchor",
    "estimated_gas": 85_000,
    "actual_gas": 82_345,
    "gas_price_wei": 30_000_000_000,
    "total_cost_wei": 2_470_350_000_000_000,
    "total_cost_usd": "0.55",
    "tx_hash": SAMPLE_TX_HASH,
    "created_at": _ts(days_ago=10),
}

GAS_COST_DEPLOY_ETHEREUM: Dict[str, Any] = {
    "cost_id": "GAS-DEPLOY-001",
    "chain": "ethereum",
    "operation": "deploy",
    "estimated_gas": 2_500_000,
    "actual_gas": 2_340_000,
    "gas_price_wei": 50_000_000_000,
    "total_cost_wei": 117_000_000_000_000_000,
    "total_cost_usd": "180.50",
    "tx_hash": SAMPLE_TX_HASH_2,
    "created_at": _ts(days_ago=90),
}

ALL_SAMPLE_GAS_COSTS: List[Dict[str, Any]] = [
    GAS_COST_ANCHOR_POLYGON, GAS_COST_DEPLOY_ETHEREUM,
]


# ---------------------------------------------------------------------------
# Sample Batch Jobs
# ---------------------------------------------------------------------------

BATCH_JOB_COMPLETED: Dict[str, Any] = {
    "job_id": "JOB-BATCH-001",
    "status": "completed",
    "total_records": 100,
    "processed_records": 100,
    "failed_records": 0,
    "anchor_ids": [f"ANC-BATCH-{i:03d}" for i in range(100)],
    "chain": "polygon",
    "operator_id": OPERATOR_ID_EU_001,
    "error_message": None,
    "started_at": _ts(hours_ago=2),
    "completed_at": _ts(hours_ago=1),
    "created_at": _ts(hours_ago=3),
    "provenance_hash": None,
}

BATCH_JOB_FAILED: Dict[str, Any] = {
    "job_id": "JOB-BATCH-002",
    "status": "failed",
    "total_records": 50,
    "processed_records": 30,
    "failed_records": 20,
    "anchor_ids": [f"ANC-FAIL-{i:03d}" for i in range(50)],
    "chain": "ethereum",
    "operator_id": OPERATOR_ID_EU_002,
    "error_message": "Gas price exceeded maximum threshold",
    "started_at": _ts(hours_ago=4),
    "completed_at": None,
    "created_at": _ts(hours_ago=5),
    "provenance_hash": None,
}

ALL_SAMPLE_BATCH_JOBS: List[Dict[str, Any]] = [
    BATCH_JOB_COMPLETED, BATCH_JOB_FAILED,
]


# ---------------------------------------------------------------------------
# Helper Factories
# ---------------------------------------------------------------------------

def make_anchor_record(
    anchor_id: Optional[str] = None,
    data_hash: Optional[str] = None,
    event_type: str = "dds_submission",
    chain: str = "polygon",
    status: str = "pending",
    priority: str = "p1_standard",
    operator_id: str = OPERATOR_ID_EU_001,
    commodity: Optional[str] = "cocoa",
    tx_hash: Optional[str] = None,
    block_number: Optional[int] = None,
    confirmations: int = 0,
    required_confirmations: Optional[int] = None,
    gas_used: Optional[int] = None,
    gas_price_wei: Optional[int] = None,
    retry_count: int = 0,
    error_message: Optional[str] = None,
    source_agent_id: Optional[str] = None,
    source_record_id: Optional[str] = None,
    merkle_root: Optional[str] = None,
    merkle_leaf_index: Optional[int] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build an anchor record dictionary for testing.

    Args:
        anchor_id: Anchor identifier (auto-generated if None).
        data_hash: SHA-256 data hash (auto-generated if None).
        event_type: Anchor event type.
        chain: Blockchain network.
        status: Anchor status.
        priority: Submission priority.
        operator_id: EUDR operator identifier.
        commodity: EUDR commodity type.
        tx_hash: Transaction hash.
        block_number: Block number.
        confirmations: Confirmations received.
        required_confirmations: Required confirmations (defaults by chain).
        gas_used: Gas consumed.
        gas_price_wei: Gas price in wei.
        retry_count: Retry attempts.
        error_message: Error message.
        source_agent_id: Source agent identifier.
        source_record_id: Source record identifier.
        merkle_root: Merkle root hash.
        merkle_leaf_index: Leaf index in Merkle tree.
        **overrides: Additional field overrides.

    Returns:
        Dict with all anchor record fields.
    """
    if required_confirmations is None:
        required_confirmations = CONFIRMATION_DEPTHS.get(chain, 32)
    record = {
        "anchor_id": anchor_id or f"ANC-{uuid.uuid4().hex[:12].upper()}",
        "data_hash": data_hash or _sha256(f"anchor-{uuid.uuid4().hex}"),
        "event_type": event_type,
        "chain": chain,
        "status": status,
        "priority": priority,
        "merkle_root": merkle_root,
        "merkle_leaf_index": merkle_leaf_index,
        "tx_hash": tx_hash,
        "block_number": block_number,
        "block_hash": None,
        "confirmations": confirmations,
        "required_confirmations": required_confirmations,
        "gas_used": gas_used,
        "gas_price_wei": gas_price_wei,
        "operator_id": operator_id,
        "commodity": commodity,
        "source_agent_id": source_agent_id,
        "source_record_id": source_record_id,
        "payload_metadata": {},
        "retry_count": retry_count,
        "error_message": error_message,
        "created_at": _ts(),
        "submitted_at": _ts() if status in ("submitted", "confirmed") else None,
        "confirmed_at": _ts() if status == "confirmed" else None,
        "expires_at": _ts(days_ago=-(365 * EUDR_RETENTION_YEARS)),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_merkle_tree(
    tree_id: Optional[str] = None,
    leaf_hashes: Optional[List[str]] = None,
    leaf_count: int = 4,
    hash_algorithm: str = "sha256",
    sorted_tree: bool = True,
    chain: Optional[str] = "polygon",
    tx_hash: Optional[str] = None,
    block_number: Optional[int] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a Merkle tree dictionary for testing.

    Args:
        tree_id: Tree identifier (auto-generated if None).
        leaf_hashes: List of leaf hashes (auto-generated if None).
        leaf_count: Number of leaves to generate (if leaf_hashes is None).
        hash_algorithm: Hash algorithm used.
        sorted_tree: Whether tree is sorted.
        chain: Blockchain network.
        tx_hash: Transaction hash for on-chain anchor.
        block_number: Block number.
        **overrides: Additional field overrides.

    Returns:
        Dict with all Merkle tree fields.
    """
    if leaf_hashes is None:
        leaf_hashes = [
            _sha256(f"leaf-{i}-{uuid.uuid4().hex[:8]}")
            for i in range(leaf_count)
        ]
    if sorted_tree:
        leaf_hashes = sorted(leaf_hashes)
    root_hash = _build_merkle_root(leaf_hashes) if leaf_hashes else None
    depth = math.ceil(math.log2(max(len(leaf_hashes), 1))) if leaf_hashes else 0
    leaves = [
        {
            "leaf_index": i,
            "data_hash": h,
            "anchor_id": f"ANC-{uuid.uuid4().hex[:8].upper()}",
            "leaf_hash": _sha256(f"leaf-{h}"),
        }
        for i, h in enumerate(leaf_hashes)
    ]
    record = {
        "tree_id": tree_id or f"TREE-{uuid.uuid4().hex[:8].upper()}",
        "root_hash": root_hash,
        "leaf_count": len(leaf_hashes),
        "leaves": leaves,
        "depth": depth,
        "hash_algorithm": hash_algorithm,
        "sorted": sorted_tree,
        "chain": chain,
        "tx_hash": tx_hash,
        "block_number": block_number,
        "anchor_ids": [leaf["anchor_id"] for leaf in leaves],
        "created_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_merkle_proof(
    proof_id: Optional[str] = None,
    tree_id: str = TREE_ID_001,
    root_hash: Optional[str] = None,
    leaf_hash: Optional[str] = None,
    leaf_index: int = 0,
    sibling_hashes: Optional[List[str]] = None,
    path_indices: Optional[List[int]] = None,
    hash_algorithm: str = "sha256",
    proof_format: str = "json",
    anchor_id: Optional[str] = None,
    verified: Optional[bool] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a Merkle proof dictionary for testing.

    Args:
        proof_id: Proof identifier (auto-generated if None).
        tree_id: Merkle tree identifier.
        root_hash: Root hash (auto-generated if None).
        leaf_hash: Leaf hash being proven (auto-generated if None).
        leaf_index: Leaf index in the tree.
        sibling_hashes: Sibling hashes in proof path.
        path_indices: Left(0)/right(1) indicators.
        hash_algorithm: Hash algorithm used.
        proof_format: Output format.
        anchor_id: Associated anchor ID.
        verified: Whether proof has been verified.
        **overrides: Additional field overrides.

    Returns:
        Dict with all Merkle proof fields.
    """
    if sibling_hashes is None:
        sibling_hashes = [_sha256(f"sibling-{i}") for i in range(2)]
    if path_indices is None:
        path_indices = [0] * len(sibling_hashes)
    record = {
        "proof_id": proof_id or f"PROOF-{uuid.uuid4().hex[:8].upper()}",
        "tree_id": tree_id,
        "root_hash": root_hash or _sha256(f"root-{uuid.uuid4().hex}"),
        "leaf_hash": leaf_hash or _sha256(f"leaf-{uuid.uuid4().hex}"),
        "leaf_index": leaf_index,
        "sibling_hashes": sibling_hashes,
        "path_indices": path_indices,
        "hash_algorithm": hash_algorithm,
        "proof_format": proof_format,
        "anchor_id": anchor_id,
        "verified": verified,
        "created_at": _ts(),
    }
    record.update(overrides)
    return record


def make_smart_contract(
    contract_id: Optional[str] = None,
    contract_type: str = "anchor_registry",
    chain: str = "polygon",
    address: Optional[str] = None,
    deployer_address: Optional[str] = None,
    version: str = "1.0.0",
    status: str = "deployed",
    abi_hash: Optional[str] = None,
    deploy_tx_hash: Optional[str] = None,
    deploy_block_number: Optional[int] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a smart contract dictionary for testing.

    Args:
        contract_id: Contract identifier (auto-generated if None).
        contract_type: Type of contract.
        chain: Blockchain network.
        address: Contract address.
        deployer_address: Deployer address.
        version: Contract version.
        status: Contract status.
        abi_hash: ABI hash.
        deploy_tx_hash: Deployment transaction hash.
        deploy_block_number: Deployment block number.
        **overrides: Additional field overrides.

    Returns:
        Dict with all smart contract fields.
    """
    record = {
        "contract_id": contract_id or f"CTR-{uuid.uuid4().hex[:8].upper()}",
        "contract_type": contract_type,
        "chain": chain,
        "address": address or f"0x{uuid.uuid4().hex[:40]}",
        "deployer_address": deployer_address or SAMPLE_DEPLOYER_ADDRESS,
        "deploy_tx_hash": deploy_tx_hash or f"0x{uuid.uuid4().hex}",
        "deploy_block_number": deploy_block_number or SAMPLE_BLOCK_NUMBER,
        "abi_hash": abi_hash or _sha256(f"abi-{contract_type}-{version}"),
        "version": version,
        "status": status,
        "created_at": _ts(days_ago=30),
        "deployed_at": _ts(days_ago=30) if status != "deploying" else None,
        "paused_at": _ts(days_ago=5) if status == "paused" else None,
        "deprecated_at": _ts(days_ago=1) if status == "deprecated" else None,
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_contract_event(
    event_id: Optional[str] = None,
    event_type: str = "anchor_created",
    contract_address: Optional[str] = None,
    chain: str = "polygon",
    tx_hash: Optional[str] = None,
    block_number: int = SAMPLE_BLOCK_NUMBER,
    log_index: int = 0,
    event_data: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a contract event dictionary for testing.

    Args:
        event_id: Event identifier (auto-generated if None).
        event_type: Type of on-chain event.
        contract_address: Emitting contract address.
        chain: Blockchain network.
        tx_hash: Transaction hash.
        block_number: Block number.
        log_index: Log index.
        event_data: Decoded event data.
        **overrides: Additional field overrides.

    Returns:
        Dict with all contract event fields.
    """
    record = {
        "event_id": event_id or f"EVT-{uuid.uuid4().hex[:8].upper()}",
        "event_type": event_type,
        "contract_address": contract_address or SAMPLE_CONTRACT_ADDRESS,
        "chain": chain,
        "tx_hash": tx_hash or f"0x{uuid.uuid4().hex}",
        "block_number": block_number,
        "block_hash": SAMPLE_BLOCK_HASH,
        "log_index": log_index,
        "event_data": event_data or {},
        "indexed_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_chain_connection(
    connection_id: Optional[str] = None,
    chain: str = "polygon",
    rpc_endpoint: Optional[str] = None,
    status: str = "connected",
    latest_block: Optional[int] = None,
    peer_count: Optional[int] = None,
    chain_id: Optional[int] = None,
    network_name: Optional[str] = None,
    error_message: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a chain connection dictionary for testing.

    Args:
        connection_id: Connection identifier (auto-generated if None).
        chain: Blockchain network.
        rpc_endpoint: RPC endpoint URL.
        status: Connection status.
        latest_block: Latest known block number.
        peer_count: Connected peer count.
        chain_id: Numeric chain ID.
        network_name: Human-readable network name.
        error_message: Error message if in error state.
        **overrides: Additional field overrides.

    Returns:
        Dict with all chain connection fields.
    """
    record = {
        "connection_id": connection_id or f"CONN-{uuid.uuid4().hex[:8].upper()}",
        "chain": chain,
        "rpc_endpoint": rpc_endpoint or RPC_ENDPOINTS.get(chain, SAMPLE_RPC_POLYGON),
        "status": status,
        "latest_block": latest_block or (SAMPLE_BLOCK_NUMBER if status == "connected" else None),
        "peer_count": peer_count or (25 if status == "connected" else None),
        "chain_id": chain_id or CHAIN_IDS.get(chain),
        "network_name": network_name or f"{chain.title()} Network",
        "connected_at": _ts(hours_ago=1) if status == "connected" else None,
        "last_heartbeat": _ts() if status == "connected" else None,
        "error_message": error_message if status in ("error", "disconnected") else None,
    }
    record.update(overrides)
    return record


def make_verification_result(
    verification_id: Optional[str] = None,
    anchor_id: str = ANCHOR_ID_001,
    status: str = "verified",
    chain: str = "polygon",
    on_chain_root: Optional[str] = None,
    computed_root: Optional[str] = None,
    data_hash_match: Optional[bool] = None,
    root_hash_match: Optional[bool] = None,
    block_number: Optional[int] = None,
    gas_used: Optional[int] = None,
    cached: bool = False,
    error_message: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a verification result dictionary for testing.

    Args:
        verification_id: Verification identifier (auto-generated if None).
        anchor_id: Anchor record being verified.
        status: Verification status.
        chain: Blockchain network.
        on_chain_root: On-chain Merkle root.
        computed_root: Locally computed root.
        data_hash_match: Whether data hash matches.
        root_hash_match: Whether root hash matches.
        block_number: Block number at verification.
        gas_used: Gas consumed.
        cached: Whether result was cached.
        error_message: Error message if failed.
        **overrides: Additional field overrides.

    Returns:
        Dict with all verification result fields.
    """
    if status == "verified":
        root = on_chain_root or SAMPLE_MERKLE_ROOT
        on_chain_root = root
        computed_root = computed_root or root
        data_hash_match = data_hash_match if data_hash_match is not None else True
        root_hash_match = root_hash_match if root_hash_match is not None else True
    elif status == "tampered":
        on_chain_root = on_chain_root or SAMPLE_MERKLE_ROOT
        computed_root = computed_root or _sha256("tampered-root")
        data_hash_match = data_hash_match if data_hash_match is not None else True
        root_hash_match = False
        error_message = error_message or "Root hash mismatch"
    elif status == "not_found":
        error_message = error_message or "Anchor not found on-chain"
    elif status == "error":
        error_message = error_message or "RPC call failed"

    record = {
        "verification_id": verification_id or f"VER-{uuid.uuid4().hex[:8].upper()}",
        "anchor_id": anchor_id,
        "status": status,
        "chain": chain,
        "merkle_proof": None,
        "on_chain_root": on_chain_root,
        "computed_root": computed_root,
        "data_hash_match": data_hash_match,
        "root_hash_match": root_hash_match,
        "block_number": block_number or (SAMPLE_BLOCK_NUMBER if status == "verified" else None),
        "gas_used": gas_used,
        "cached": cached,
        "error_message": error_message,
        "verified_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_access_grant(
    grant_id: Optional[str] = None,
    anchor_id: str = ANCHOR_ID_001,
    grantor_id: str = OPERATOR_ID_EU_001,
    grantee_id: str = "CA-DE-BMEL-001",
    access_level: str = "competent_authority",
    status: str = "active",
    scope: Optional[Dict[str, Any]] = None,
    multi_party_confirmations: int = 2,
    required_confirmations: int = 2,
    expires_in_days: int = 365,
    revocation_reason: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build an access grant dictionary for testing.

    Args:
        grant_id: Grant identifier (auto-generated if None).
        anchor_id: Anchor record being shared.
        grantor_id: Data owner operator ID.
        grantee_id: Receiving party identifier.
        access_level: Access level granted.
        status: Grant status.
        scope: Optional scope restrictions.
        multi_party_confirmations: Confirmations received.
        required_confirmations: Confirmations required.
        expires_in_days: Days until expiry (from now).
        revocation_reason: Reason for revocation.
        **overrides: Additional field overrides.

    Returns:
        Dict with all access grant fields.
    """
    now = _ts()
    record = {
        "grant_id": grant_id or f"GRANT-{uuid.uuid4().hex[:8].upper()}",
        "anchor_id": anchor_id,
        "grantor_id": grantor_id,
        "grantee_id": grantee_id,
        "access_level": access_level,
        "status": status,
        "scope": scope,
        "multi_party_confirmations": multi_party_confirmations,
        "required_confirmations": required_confirmations,
        "granted_at": now,
        "expires_at": _ts(days_ago=-expires_in_days) if status != "expired" else _ts(days_ago=10),
        "revoked_at": _ts() if status == "revoked" else None,
        "revocation_reason": revocation_reason if status == "revoked" else None,
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_evidence_package(
    package_id: Optional[str] = None,
    anchor_ids: Optional[List[str]] = None,
    fmt: str = "json",
    operator_id: str = OPERATOR_ID_EU_001,
    signed: bool = False,
    signature: Optional[str] = None,
    signer_id: Optional[str] = None,
    package_hash: Optional[str] = None,
    retention_years: int = EUDR_RETENTION_YEARS,
    chain_references: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build an evidence package dictionary for testing.

    Args:
        package_id: Package identifier (auto-generated if None).
        anchor_ids: List of anchor IDs (auto-generated if None).
        fmt: Evidence format (json/pdf/eudr_xml).
        operator_id: EUDR operator identifier.
        signed: Whether package is signed.
        signature: Digital signature.
        signer_id: Signing key identifier.
        package_hash: SHA-256 hash of package.
        retention_years: Retention period in years.
        chain_references: On-chain references.
        **overrides: Additional field overrides.

    Returns:
        Dict with all evidence package fields.
    """
    pkg_id = package_id or f"PKG-{uuid.uuid4().hex[:8].upper()}"
    ids = anchor_ids or [ANCHOR_ID_001]
    record = {
        "package_id": pkg_id,
        "anchor_ids": ids,
        "format": fmt,
        "operator_id": operator_id,
        "merkle_proofs": [],
        "verification_results": [],
        "chain_references": chain_references or {"chain": "polygon"},
        "package_hash": package_hash or _sha256(f"evidence-{pkg_id}"),
        "signed": signed,
        "signature": signature if signed else None,
        "signer_id": signer_id if signed else None,
        "retention_until": _ts(days_ago=-(365 * retention_years)),
        "created_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_gas_cost(
    cost_id: Optional[str] = None,
    chain: str = "polygon",
    operation: str = "anchor",
    estimated_gas: int = 85_000,
    actual_gas: Optional[int] = None,
    gas_price_wei: int = 30_000_000_000,
    tx_hash: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a gas cost dictionary for testing.

    Args:
        cost_id: Cost record identifier (auto-generated if None).
        chain: Blockchain network.
        operation: Type of operation.
        estimated_gas: Estimated gas units.
        actual_gas: Actual gas consumed.
        gas_price_wei: Gas price in wei.
        tx_hash: Associated transaction hash.
        **overrides: Additional field overrides.

    Returns:
        Dict with all gas cost fields.
    """
    ag = actual_gas or estimated_gas
    total_wei = ag * gas_price_wei
    record = {
        "cost_id": cost_id or f"GAS-{uuid.uuid4().hex[:8].upper()}",
        "chain": chain,
        "operation": operation,
        "estimated_gas": estimated_gas,
        "actual_gas": actual_gas,
        "gas_price_wei": gas_price_wei,
        "total_cost_wei": total_wei,
        "total_cost_usd": None,
        "tx_hash": tx_hash,
        "created_at": _ts(),
    }
    record.update(overrides)
    return record


def make_batch_job(
    job_id: Optional[str] = None,
    status: str = "queued",
    total_records: int = 10,
    processed_records: int = 0,
    failed_records: int = 0,
    chain: str = "polygon",
    operator_id: str = OPERATOR_ID_EU_001,
    error_message: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a batch job dictionary for testing.

    Args:
        job_id: Job identifier (auto-generated if None).
        status: Job status.
        total_records: Total records to process.
        processed_records: Records processed so far.
        failed_records: Records that failed.
        chain: Target blockchain network.
        operator_id: EUDR operator identifier.
        error_message: Error message if failed.
        **overrides: Additional field overrides.

    Returns:
        Dict with all batch job fields.
    """
    record = {
        "job_id": job_id or f"JOB-{uuid.uuid4().hex[:8].upper()}",
        "status": status,
        "total_records": total_records,
        "processed_records": processed_records,
        "failed_records": failed_records,
        "anchor_ids": [f"ANC-{uuid.uuid4().hex[:8].upper()}" for _ in range(total_records)],
        "chain": chain,
        "operator_id": operator_id,
        "error_message": error_message,
        "started_at": _ts() if status in ("processing", "completed", "failed") else None,
        "completed_at": _ts() if status == "completed" else None,
        "created_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Assertion Helpers
# ---------------------------------------------------------------------------

def assert_valid_sha256(hash_str: str) -> None:
    """Assert that a string is a valid SHA-256 hex digest.

    Args:
        hash_str: The hash string to validate.

    Raises:
        AssertionError: If hash is not a valid 64-char hex string.
    """
    assert isinstance(hash_str, str), f"Hash must be string, got {type(hash_str)}"
    assert len(hash_str) == SHA256_HEX_LENGTH, (
        f"SHA-256 hash length must be {SHA256_HEX_LENGTH}, got {len(hash_str)}"
    )
    assert all(c in "0123456789abcdef" for c in hash_str.lower()), (
        "Hash must be lowercase hex characters only"
    )


def assert_valid_tx_hash(tx_hash: str) -> None:
    """Assert that a string looks like a valid transaction hash.

    Args:
        tx_hash: The transaction hash to validate.

    Raises:
        AssertionError: If tx_hash does not look like a valid hash.
    """
    assert isinstance(tx_hash, str), f"tx_hash must be string, got {type(tx_hash)}"
    assert len(tx_hash) > 0, "tx_hash must not be empty"
    if tx_hash.startswith("0x"):
        hex_part = tx_hash[2:]
        assert len(hex_part) > 0, "tx_hash 0x prefix must be followed by hex chars"
        assert all(c in "0123456789abcdef" for c in hex_part.lower()), (
            "tx_hash must be hex characters after 0x prefix"
        )


def assert_valid_provenance_hash(hash_value: str) -> None:
    """Assert that a provenance hash is a valid SHA-256 hex digest.

    Args:
        hash_value: The hash string to validate.

    Raises:
        AssertionError: If hash is not a valid 64-char hex string.
    """
    assert_valid_sha256(hash_value)


def assert_anchor_valid(record: Dict[str, Any]) -> None:
    """Assert that an anchor record has all required fields and valid values.

    Args:
        record: Anchor record dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "anchor_id" in record, "Missing anchor_id"
    assert "data_hash" in record, "Missing data_hash"
    assert len(record["data_hash"]) >= SHA256_HEX_LENGTH, (
        f"data_hash must be at least {SHA256_HEX_LENGTH} chars"
    )
    assert "event_type" in record, "Missing event_type"
    assert record["event_type"] in ANCHOR_EVENT_TYPES, (
        f"Invalid event_type: {record['event_type']}"
    )
    assert "chain" in record, "Missing chain"
    assert record["chain"] in BLOCKCHAIN_NETWORKS, (
        f"Invalid chain: {record['chain']}"
    )
    assert "status" in record, "Missing status"
    assert record["status"] in ANCHOR_STATUSES, (
        f"Invalid status: {record['status']}"
    )
    assert "priority" in record, "Missing priority"
    assert record["priority"] in ANCHOR_PRIORITIES, (
        f"Invalid priority: {record['priority']}"
    )
    assert "operator_id" in record, "Missing operator_id"
    assert len(record["operator_id"]) > 0, "operator_id must not be empty"


def assert_merkle_tree_valid(tree: Dict[str, Any]) -> None:
    """Assert that a Merkle tree dictionary has valid structure.

    Args:
        tree: Merkle tree dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "tree_id" in tree, "Missing tree_id"
    assert "root_hash" in tree, "Missing root_hash"
    assert len(tree["root_hash"]) >= SHA256_HEX_LENGTH, (
        f"root_hash must be at least {SHA256_HEX_LENGTH} chars"
    )
    assert "leaf_count" in tree, "Missing leaf_count"
    assert tree["leaf_count"] >= 1, "leaf_count must be >= 1"
    assert "depth" in tree, "Missing depth"
    assert tree["depth"] >= 0, "depth must be >= 0"
    assert "hash_algorithm" in tree, "Missing hash_algorithm"
    assert tree["hash_algorithm"] in HASH_ALGORITHMS, (
        f"Invalid hash_algorithm: {tree['hash_algorithm']}"
    )


def assert_merkle_proof_valid(proof: Dict[str, Any]) -> None:
    """Assert that a Merkle proof dictionary has valid structure.

    Args:
        proof: Merkle proof dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "proof_id" in proof, "Missing proof_id"
    assert "tree_id" in proof, "Missing tree_id"
    assert "root_hash" in proof, "Missing root_hash"
    assert len(proof["root_hash"]) >= SHA256_HEX_LENGTH, (
        f"root_hash must be at least {SHA256_HEX_LENGTH} chars"
    )
    assert "leaf_hash" in proof, "Missing leaf_hash"
    assert len(proof["leaf_hash"]) >= SHA256_HEX_LENGTH, (
        f"leaf_hash must be at least {SHA256_HEX_LENGTH} chars"
    )
    assert "leaf_index" in proof, "Missing leaf_index"
    assert proof["leaf_index"] >= 0, "leaf_index must be >= 0"
    assert "sibling_hashes" in proof, "Missing sibling_hashes"
    assert isinstance(proof["sibling_hashes"], list), "sibling_hashes must be list"
    assert "path_indices" in proof, "Missing path_indices"
    assert isinstance(proof["path_indices"], list), "path_indices must be list"
    assert len(proof["sibling_hashes"]) == len(proof["path_indices"]), (
        "sibling_hashes and path_indices must have equal length"
    )


def assert_contract_valid(contract: Dict[str, Any]) -> None:
    """Assert that a smart contract dictionary has valid structure.

    Args:
        contract: Smart contract dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "contract_id" in contract, "Missing contract_id"
    assert "contract_type" in contract, "Missing contract_type"
    assert contract["contract_type"] in CONTRACT_TYPES, (
        f"Invalid contract_type: {contract['contract_type']}"
    )
    assert "chain" in contract, "Missing chain"
    assert contract["chain"] in BLOCKCHAIN_NETWORKS, (
        f"Invalid chain: {contract['chain']}"
    )
    assert "status" in contract, "Missing status"
    assert contract["status"] in CONTRACT_STATUSES, (
        f"Invalid status: {contract['status']}"
    )
    assert "version" in contract, "Missing version"


def assert_verification_valid(result: Dict[str, Any]) -> None:
    """Assert that a verification result has valid structure.

    Args:
        result: Verification result dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "verification_id" in result, "Missing verification_id"
    assert "anchor_id" in result, "Missing anchor_id"
    assert "status" in result, "Missing status"
    assert result["status"] in VERIFICATION_STATUSES, (
        f"Invalid status: {result['status']}"
    )
    assert "chain" in result, "Missing chain"
    assert result["chain"] in BLOCKCHAIN_NETWORKS, (
        f"Invalid chain: {result['chain']}"
    )
    if result["status"] == "verified":
        assert result.get("root_hash_match") is True, (
            "Verified result must have root_hash_match=True"
        )


def assert_access_grant_valid(grant: Dict[str, Any]) -> None:
    """Assert that an access grant has valid structure.

    Args:
        grant: Access grant dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "grant_id" in grant, "Missing grant_id"
    assert "anchor_id" in grant, "Missing anchor_id"
    assert "grantor_id" in grant, "Missing grantor_id"
    assert "grantee_id" in grant, "Missing grantee_id"
    assert "access_level" in grant, "Missing access_level"
    assert grant["access_level"] in ACCESS_LEVELS, (
        f"Invalid access_level: {grant['access_level']}"
    )
    assert "status" in grant, "Missing status"
    assert grant["status"] in ACCESS_STATUSES, (
        f"Invalid status: {grant['status']}"
    )


def assert_evidence_package_valid(package: Dict[str, Any]) -> None:
    """Assert that an evidence package has valid structure.

    Args:
        package: Evidence package dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "package_id" in package, "Missing package_id"
    assert "anchor_ids" in package, "Missing anchor_ids"
    assert isinstance(package["anchor_ids"], list), "anchor_ids must be list"
    assert len(package["anchor_ids"]) >= 1, "anchor_ids must not be empty"
    assert "format" in package, "Missing format"
    assert package["format"] in EVIDENCE_FORMATS, (
        f"Invalid format: {package['format']}"
    )
    assert "operator_id" in package, "Missing operator_id"
    assert "package_hash" in package, "Missing package_hash"


def assert_gas_cost_valid(cost: Dict[str, Any]) -> None:
    """Assert that a gas cost record has valid structure.

    Args:
        cost: Gas cost dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "cost_id" in cost, "Missing cost_id"
    assert "chain" in cost, "Missing chain"
    assert cost["chain"] in BLOCKCHAIN_NETWORKS, (
        f"Invalid chain: {cost['chain']}"
    )
    assert "operation" in cost, "Missing operation"
    assert len(cost["operation"]) > 0, "operation must not be empty"


def assert_contract_event_valid(event: Dict[str, Any]) -> None:
    """Assert that a contract event has valid structure.

    Args:
        event: Contract event dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "event_id" in event, "Missing event_id"
    assert "event_type" in event, "Missing event_type"
    assert event["event_type"] in EVENT_TYPES, (
        f"Invalid event_type: {event['event_type']}"
    )
    assert "contract_address" in event, "Missing contract_address"
    assert "chain" in event, "Missing chain"
    assert event["chain"] in BLOCKCHAIN_NETWORKS, (
        f"Invalid chain: {event['chain']}"
    )
    assert "tx_hash" in event, "Missing tx_hash"
    assert "block_number" in event, "Missing block_number"
    assert event["block_number"] >= 0, "block_number must be >= 0"


def assert_chain_connection_valid(conn: Dict[str, Any]) -> None:
    """Assert that a chain connection has valid structure.

    Args:
        conn: Chain connection dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "connection_id" in conn, "Missing connection_id"
    assert "chain" in conn, "Missing chain"
    assert conn["chain"] in BLOCKCHAIN_NETWORKS, (
        f"Invalid chain: {conn['chain']}"
    )
    assert "rpc_endpoint" in conn, "Missing rpc_endpoint"
    assert len(conn["rpc_endpoint"]) > 0, "rpc_endpoint must not be empty"
    assert "status" in conn, "Missing status"
    assert conn["status"] in CONNECTION_STATUSES, (
        f"Invalid status: {conn['status']}"
    )


# ---------------------------------------------------------------------------
# Configuration Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def bci_config() -> Dict[str, Any]:
    """Create a BlockchainIntegrationConfig-compatible dictionary with test defaults."""
    return {
        "database_url": "postgresql://localhost:5432/greenlang_test",
        "redis_url": "redis://localhost:6379/2",
        "log_level": "DEBUG",
        "pool_size": 5,
        "batch_size": 100,
        "batch_interval_s": 60,
        "max_retries": 3,
        "retry_backoff_factor": 2.0,
        "anchor_priority_levels": 3,
        "gas_price_multiplier": 1.2,
        "default_gas_limit": 300_000,
        "contract_deploy_timeout_s": 120,
        "abi_cache_enabled": True,
        "primary_chain": "polygon",
        "fallback_chain": "ethereum",
        "confirmation_depth_ethereum": 12,
        "confirmation_depth_polygon": 32,
        "confirmation_depth_fabric": 1,
        "confirmation_depth_besu": 1,
        "rpc_timeout_s": 30,
        "max_connections_per_chain": 5,
        "verification_cache_ttl_s": 300,
        "max_batch_verify_size": 100,
        "proof_format": "json",
        "polling_interval_s": 15,
        "max_events_per_poll": 100,
        "reorg_depth": 64,
        "webhook_timeout_s": 10,
        "max_tree_leaves": 10_000,
        "sorted_tree": True,
        "hash_algorithm": "sha256",
        "max_grants_per_record": 50,
        "grant_expiry_days": 365,
        "require_multi_party_confirmation": True,
        "min_confirmations": 2,
        "evidence_formats": list(EVIDENCE_FORMATS),
        "evidence_retention_years": EUDR_RETENTION_YEARS,
        "package_signing_enabled": True,
        "batch_max_size": MAX_BATCH_SIZE,
        "batch_concurrency": 4,
        "batch_timeout_s": 300,
        "retention_years": EUDR_RETENTION_YEARS,
        "enable_provenance": True,
        "genesis_hash": "GL-EUDR-BCI-013-TEST-GENESIS",
        "enable_metrics": False,
        "rate_limit": 300,
    }


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset any singleton state between tests to prevent cross-test contamination."""
    yield
    try:
        from greenlang.agents.eudr.blockchain_integration.config import reset_config
        reset_config()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Engine Fixtures (with graceful pytest.skip for unimplemented)
# ---------------------------------------------------------------------------

@pytest.fixture
def anchor_engine(bci_config):
    """Create a TransactionAnchor engine instance for testing."""
    try:
        from greenlang.agents.eudr.blockchain_integration.transaction_anchor import (
            TransactionAnchor,
        )
        return TransactionAnchor()
    except ImportError:
        pytest.skip("TransactionAnchor not yet implemented")


@pytest.fixture
def contract_engine(bci_config):
    """Create a SmartContractManager engine instance for testing."""
    try:
        from greenlang.agents.eudr.blockchain_integration.smart_contract_manager import (
            SmartContractManager,
        )
        return SmartContractManager()
    except ImportError:
        pytest.skip("SmartContractManager not yet implemented")


@pytest.fixture
def chain_engine(bci_config):
    """Create a MultiChainConnector engine instance for testing."""
    try:
        from greenlang.agents.eudr.blockchain_integration.multi_chain_connector import (
            MultiChainConnector,
        )
        return MultiChainConnector()
    except ImportError:
        pytest.skip("MultiChainConnector not yet implemented")


@pytest.fixture
def verification_engine(bci_config):
    """Create a VerificationEngine instance for testing."""
    try:
        from greenlang.agents.eudr.blockchain_integration.verification_engine import (
            VerificationEngine,
        )
        return VerificationEngine()
    except ImportError:
        pytest.skip("VerificationEngine not yet implemented")


@pytest.fixture
def event_engine(bci_config):
    """Create an EventListener engine instance for testing."""
    try:
        from greenlang.agents.eudr.blockchain_integration.event_listener import (
            EventListener,
        )
        return EventListener()
    except ImportError:
        pytest.skip("EventListener not yet implemented")


@pytest.fixture
def merkle_engine(bci_config):
    """Create a MerkleProofGenerator engine instance for testing."""
    try:
        from greenlang.agents.eudr.blockchain_integration.merkle_proof_generator import (
            MerkleProofGenerator,
        )
        return MerkleProofGenerator()
    except ImportError:
        pytest.skip("MerkleProofGenerator not yet implemented")


@pytest.fixture
def sharing_engine(bci_config):
    """Create a CrossPartySharing engine instance for testing."""
    try:
        from greenlang.agents.eudr.blockchain_integration.cross_party_sharing import (
            CrossPartySharing,
        )
        return CrossPartySharing()
    except ImportError:
        pytest.skip("CrossPartySharing not yet implemented")


@pytest.fixture
def evidence_engine(bci_config):
    """Create a ComplianceEvidencePackager engine instance for testing."""
    try:
        from greenlang.agents.eudr.blockchain_integration.compliance_evidence_packager import (
            ComplianceEvidencePackager,
        )
        return ComplianceEvidencePackager()
    except ImportError:
        pytest.skip("ComplianceEvidencePackager not yet implemented")


@pytest.fixture
def blockchain_service(bci_config):
    """Create the top-level BlockchainIntegrationService facade for testing."""
    try:
        from greenlang.agents.eudr.blockchain_integration.setup import (
            BlockchainIntegrationService,
        )
        return BlockchainIntegrationService()
    except ImportError:
        pytest.skip("BlockchainIntegrationService not yet implemented")


# ---------------------------------------------------------------------------
# Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_anchor() -> Dict[str, Any]:
    """Return a sample confirmed DDS anchor on Polygon."""
    return copy.deepcopy(ANCHOR_DDS_POLYGON)


@pytest.fixture
def sample_anchor_pending() -> Dict[str, Any]:
    """Return a sample pending batch anchor on Fabric."""
    return copy.deepcopy(ANCHOR_BATCH_FABRIC)


@pytest.fixture
def sample_merkle_tree() -> Dict[str, Any]:
    """Return a sample 4-leaf Merkle tree."""
    return copy.deepcopy(MERKLE_TREE_4_LEAVES)


@pytest.fixture
def sample_merkle_proof() -> Dict[str, Any]:
    """Return a sample Merkle proof for leaf 0."""
    return copy.deepcopy(MERKLE_PROOF_LEAF_0)


@pytest.fixture
def sample_contract() -> Dict[str, Any]:
    """Return a sample deployed anchor registry contract."""
    return copy.deepcopy(CONTRACT_ANCHOR_REGISTRY_POLYGON)


@pytest.fixture
def sample_event() -> Dict[str, Any]:
    """Return a sample anchor_created event."""
    return copy.deepcopy(EVENT_ANCHOR_CREATED)


@pytest.fixture
def sample_connection() -> Dict[str, Any]:
    """Return a sample connected Polygon chain connection."""
    return copy.deepcopy(CONNECTION_POLYGON)


@pytest.fixture
def sample_verification() -> Dict[str, Any]:
    """Return a sample verified verification result."""
    return copy.deepcopy(VERIFICATION_VERIFIED)


@pytest.fixture
def sample_grant() -> Dict[str, Any]:
    """Return a sample active competent authority access grant."""
    return copy.deepcopy(GRANT_AUTHORITY_ACTIVE)


@pytest.fixture
def sample_evidence() -> Dict[str, Any]:
    """Return a sample JSON evidence package."""
    return copy.deepcopy(EVIDENCE_JSON_DDS)


@pytest.fixture
def sample_gas_cost() -> Dict[str, Any]:
    """Return a sample anchor gas cost on Polygon."""
    return copy.deepcopy(GAS_COST_ANCHOR_POLYGON)


@pytest.fixture
def sample_batch_job() -> Dict[str, Any]:
    """Return a sample completed batch job."""
    return copy.deepcopy(BATCH_JOB_COMPLETED)
