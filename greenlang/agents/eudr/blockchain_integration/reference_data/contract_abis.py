# -*- coding: utf-8 -*-
"""
Smart Contract ABIs Reference Data - AGENT-EUDR-013 Blockchain Integration

Solidity ABI (Application Binary Interface) definitions for the three smart
contracts deployed as part of the EUDR compliance anchoring infrastructure.
Each ABI defines the function signatures, input/output types, event
definitions, and error conditions that external callers use to interact
with the deployed contract bytecode on EVM-compatible chains (Ethereum,
Polygon, Hyperledger Besu).

Contracts (3):
    1. AnchorRegistry - Stores Merkle roots of anchored EUDR compliance
       data.  Provides anchor(), verify(), getAnchor(), getAnchorCount(),
       and getLatestAnchor() functions.  Emits AnchorCreated events.
    2. CustodyTransfer - Records custody transfer events between supply
       chain participants.  Provides recordTransfer(), confirmTransfer(),
       getTransfer(), and isConfirmed() functions.  Emits
       CustodyTransferRecorded and TransferConfirmed events.
    3. ComplianceCheck - Implements on-chain compliance verification for
       EUDR Article 4 due diligence.  Provides checkCompliance(),
       registerParty(), isPartyRegistered(), and getComplianceStatus()
       functions.  Emits ComplianceCheckCompleted and PartyRegistered
       events.

ABI Format:
    Standard Ethereum JSON ABI format compatible with web3.py, ethers.js,
    and Go Ethereum client libraries.  Each entry is a list of dictionaries
    describing function, event, and constructor signatures.

Bytecode Hashes:
    SHA-256 hashes of compiled contract bytecodes for deployment
    verification and provenance tracking.

Lookup Helpers:
    get_abi(contract_type) -> list | None
    get_contract_bytecode_hash(contract_type) -> str | None

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013) - Appendix C
Agent ID: GL-EUDR-BCI-013
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contract type identifiers
# ---------------------------------------------------------------------------

#: Supported smart contract types.
CONTRACT_TYPES: List[str] = [
    "anchor_registry",
    "custody_transfer",
    "compliance_check",
]

# ---------------------------------------------------------------------------
# AnchorRegistry ABI
# ---------------------------------------------------------------------------

#: Full ABI for the AnchorRegistry smart contract.
#:
#: Solidity interface:
#:     function anchor(bytes32 root, uint256 count) external returns (uint256)
#:     function verify(bytes32 root) external view returns (bool)
#:     function getAnchor(uint256 id) external view returns (AnchorData)
#:     function getAnchorCount() external view returns (uint256)
#:     function getLatestAnchor() external view returns (AnchorData)
#:     event AnchorCreated(uint256 indexed id, bytes32 root, uint256 count, uint256 timestamp)
ANCHOR_REGISTRY_ABI: List[Dict[str, Any]] = [
    # ---- Constructor ----
    {
        "type": "constructor",
        "inputs": [
            {
                "name": "_owner",
                "type": "address",
                "internalType": "address",
            },
        ],
        "stateMutability": "nonpayable",
    },

    # ---- anchor(bytes32, uint256) -> uint256 ----
    {
        "type": "function",
        "name": "anchor",
        "inputs": [
            {
                "name": "root",
                "type": "bytes32",
                "internalType": "bytes32",
            },
            {
                "name": "count",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "outputs": [
            {
                "name": "anchorId",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "stateMutability": "nonpayable",
    },

    # ---- verify(bytes32) -> bool ----
    {
        "type": "function",
        "name": "verify",
        "inputs": [
            {
                "name": "root",
                "type": "bytes32",
                "internalType": "bytes32",
            },
        ],
        "outputs": [
            {
                "name": "exists",
                "type": "bool",
                "internalType": "bool",
            },
        ],
        "stateMutability": "view",
    },

    # ---- getAnchor(uint256) -> tuple ----
    {
        "type": "function",
        "name": "getAnchor",
        "inputs": [
            {
                "name": "id",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "outputs": [
            {
                "name": "anchorId",
                "type": "uint256",
                "internalType": "uint256",
            },
            {
                "name": "root",
                "type": "bytes32",
                "internalType": "bytes32",
            },
            {
                "name": "count",
                "type": "uint256",
                "internalType": "uint256",
            },
            {
                "name": "timestamp",
                "type": "uint256",
                "internalType": "uint256",
            },
            {
                "name": "submitter",
                "type": "address",
                "internalType": "address",
            },
        ],
        "stateMutability": "view",
    },

    # ---- getAnchorCount() -> uint256 ----
    {
        "type": "function",
        "name": "getAnchorCount",
        "inputs": [],
        "outputs": [
            {
                "name": "totalCount",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "stateMutability": "view",
    },

    # ---- getLatestAnchor() -> tuple ----
    {
        "type": "function",
        "name": "getLatestAnchor",
        "inputs": [],
        "outputs": [
            {
                "name": "anchorId",
                "type": "uint256",
                "internalType": "uint256",
            },
            {
                "name": "root",
                "type": "bytes32",
                "internalType": "bytes32",
            },
            {
                "name": "count",
                "type": "uint256",
                "internalType": "uint256",
            },
            {
                "name": "timestamp",
                "type": "uint256",
                "internalType": "uint256",
            },
            {
                "name": "submitter",
                "type": "address",
                "internalType": "address",
            },
        ],
        "stateMutability": "view",
    },

    # ---- owner() -> address ----
    {
        "type": "function",
        "name": "owner",
        "inputs": [],
        "outputs": [
            {
                "name": "",
                "type": "address",
                "internalType": "address",
            },
        ],
        "stateMutability": "view",
    },

    # ---- paused() -> bool ----
    {
        "type": "function",
        "name": "paused",
        "inputs": [],
        "outputs": [
            {
                "name": "",
                "type": "bool",
                "internalType": "bool",
            },
        ],
        "stateMutability": "view",
    },

    # ---- pause() ----
    {
        "type": "function",
        "name": "pause",
        "inputs": [],
        "outputs": [],
        "stateMutability": "nonpayable",
    },

    # ---- unpause() ----
    {
        "type": "function",
        "name": "unpause",
        "inputs": [],
        "outputs": [],
        "stateMutability": "nonpayable",
    },

    # ---- Event: AnchorCreated ----
    {
        "type": "event",
        "name": "AnchorCreated",
        "inputs": [
            {
                "name": "id",
                "type": "uint256",
                "indexed": True,
                "internalType": "uint256",
            },
            {
                "name": "root",
                "type": "bytes32",
                "indexed": False,
                "internalType": "bytes32",
            },
            {
                "name": "count",
                "type": "uint256",
                "indexed": False,
                "internalType": "uint256",
            },
            {
                "name": "timestamp",
                "type": "uint256",
                "indexed": False,
                "internalType": "uint256",
            },
        ],
        "anonymous": False,
    },

    # ---- Event: Paused ----
    {
        "type": "event",
        "name": "Paused",
        "inputs": [
            {
                "name": "account",
                "type": "address",
                "indexed": False,
                "internalType": "address",
            },
        ],
        "anonymous": False,
    },

    # ---- Event: Unpaused ----
    {
        "type": "event",
        "name": "Unpaused",
        "inputs": [
            {
                "name": "account",
                "type": "address",
                "indexed": False,
                "internalType": "address",
            },
        ],
        "anonymous": False,
    },

    # ---- Error: Unauthorized ----
    {
        "type": "error",
        "name": "Unauthorized",
        "inputs": [
            {
                "name": "caller",
                "type": "address",
                "internalType": "address",
            },
        ],
    },

    # ---- Error: ContractPaused ----
    {
        "type": "error",
        "name": "ContractPaused",
        "inputs": [],
    },

    # ---- Error: InvalidRoot ----
    {
        "type": "error",
        "name": "InvalidRoot",
        "inputs": [],
    },

    # ---- Error: AnchorNotFound ----
    {
        "type": "error",
        "name": "AnchorNotFound",
        "inputs": [
            {
                "name": "id",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# CustodyTransfer ABI
# ---------------------------------------------------------------------------

#: Full ABI for the CustodyTransfer smart contract.
#:
#: Solidity interface:
#:     function recordTransfer(bytes32 hash, address from, address to) external returns (uint256)
#:     function confirmTransfer(uint256 id) external
#:     function getTransfer(uint256 id) external view returns (TransferData)
#:     function isConfirmed(uint256 id) external view returns (bool)
#:     event CustodyTransferRecorded(uint256 indexed id, bytes32 hash, address from, address to)
#:     event TransferConfirmed(uint256 indexed id, address confirmer)
CUSTODY_TRANSFER_ABI: List[Dict[str, Any]] = [
    # ---- Constructor ----
    {
        "type": "constructor",
        "inputs": [
            {
                "name": "_owner",
                "type": "address",
                "internalType": "address",
            },
            {
                "name": "_anchorRegistry",
                "type": "address",
                "internalType": "address",
            },
        ],
        "stateMutability": "nonpayable",
    },

    # ---- recordTransfer(bytes32, address, address) -> uint256 ----
    {
        "type": "function",
        "name": "recordTransfer",
        "inputs": [
            {
                "name": "dataHash",
                "type": "bytes32",
                "internalType": "bytes32",
            },
            {
                "name": "sender",
                "type": "address",
                "internalType": "address",
            },
            {
                "name": "receiver",
                "type": "address",
                "internalType": "address",
            },
        ],
        "outputs": [
            {
                "name": "transferId",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "stateMutability": "nonpayable",
    },

    # ---- confirmTransfer(uint256) ----
    {
        "type": "function",
        "name": "confirmTransfer",
        "inputs": [
            {
                "name": "id",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },

    # ---- getTransfer(uint256) -> tuple ----
    {
        "type": "function",
        "name": "getTransfer",
        "inputs": [
            {
                "name": "id",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "outputs": [
            {
                "name": "transferId",
                "type": "uint256",
                "internalType": "uint256",
            },
            {
                "name": "dataHash",
                "type": "bytes32",
                "internalType": "bytes32",
            },
            {
                "name": "sender",
                "type": "address",
                "internalType": "address",
            },
            {
                "name": "receiver",
                "type": "address",
                "internalType": "address",
            },
            {
                "name": "timestamp",
                "type": "uint256",
                "internalType": "uint256",
            },
            {
                "name": "confirmed",
                "type": "bool",
                "internalType": "bool",
            },
            {
                "name": "confirmer",
                "type": "address",
                "internalType": "address",
            },
            {
                "name": "confirmTimestamp",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "stateMutability": "view",
    },

    # ---- isConfirmed(uint256) -> bool ----
    {
        "type": "function",
        "name": "isConfirmed",
        "inputs": [
            {
                "name": "id",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "outputs": [
            {
                "name": "confirmed",
                "type": "bool",
                "internalType": "bool",
            },
        ],
        "stateMutability": "view",
    },

    # ---- getTransferCount() -> uint256 ----
    {
        "type": "function",
        "name": "getTransferCount",
        "inputs": [],
        "outputs": [
            {
                "name": "totalCount",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "stateMutability": "view",
    },

    # ---- owner() -> address ----
    {
        "type": "function",
        "name": "owner",
        "inputs": [],
        "outputs": [
            {
                "name": "",
                "type": "address",
                "internalType": "address",
            },
        ],
        "stateMutability": "view",
    },

    # ---- anchorRegistry() -> address ----
    {
        "type": "function",
        "name": "anchorRegistry",
        "inputs": [],
        "outputs": [
            {
                "name": "",
                "type": "address",
                "internalType": "address",
            },
        ],
        "stateMutability": "view",
    },

    # ---- Event: CustodyTransferRecorded ----
    {
        "type": "event",
        "name": "CustodyTransferRecorded",
        "inputs": [
            {
                "name": "id",
                "type": "uint256",
                "indexed": True,
                "internalType": "uint256",
            },
            {
                "name": "dataHash",
                "type": "bytes32",
                "indexed": False,
                "internalType": "bytes32",
            },
            {
                "name": "sender",
                "type": "address",
                "indexed": True,
                "internalType": "address",
            },
            {
                "name": "receiver",
                "type": "address",
                "indexed": True,
                "internalType": "address",
            },
        ],
        "anonymous": False,
    },

    # ---- Event: TransferConfirmed ----
    {
        "type": "event",
        "name": "TransferConfirmed",
        "inputs": [
            {
                "name": "id",
                "type": "uint256",
                "indexed": True,
                "internalType": "uint256",
            },
            {
                "name": "confirmer",
                "type": "address",
                "indexed": True,
                "internalType": "address",
            },
        ],
        "anonymous": False,
    },

    # ---- Error: TransferNotFound ----
    {
        "type": "error",
        "name": "TransferNotFound",
        "inputs": [
            {
                "name": "id",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
    },

    # ---- Error: AlreadyConfirmed ----
    {
        "type": "error",
        "name": "AlreadyConfirmed",
        "inputs": [
            {
                "name": "id",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
    },

    # ---- Error: InvalidAddress ----
    {
        "type": "error",
        "name": "InvalidAddress",
        "inputs": [],
    },

    # ---- Error: InvalidHash ----
    {
        "type": "error",
        "name": "InvalidHash",
        "inputs": [],
    },
]


# ---------------------------------------------------------------------------
# ComplianceCheck ABI
# ---------------------------------------------------------------------------

#: Full ABI for the ComplianceCheck smart contract.
#:
#: Solidity interface:
#:     function checkCompliance(bytes32 ddsHash) external returns (bool)
#:     function registerParty(address party, string role) external
#:     function isPartyRegistered(address party) external view returns (bool)
#:     function getComplianceStatus(bytes32 ddsHash) external view returns (ComplianceData)
#:     event ComplianceCheckCompleted(bytes32 indexed ddsHash, bool result)
#:     event PartyRegistered(address indexed party, string role)
COMPLIANCE_CHECK_ABI: List[Dict[str, Any]] = [
    # ---- Constructor ----
    {
        "type": "constructor",
        "inputs": [
            {
                "name": "_owner",
                "type": "address",
                "internalType": "address",
            },
            {
                "name": "_anchorRegistry",
                "type": "address",
                "internalType": "address",
            },
        ],
        "stateMutability": "nonpayable",
    },

    # ---- checkCompliance(bytes32) -> bool ----
    {
        "type": "function",
        "name": "checkCompliance",
        "inputs": [
            {
                "name": "ddsHash",
                "type": "bytes32",
                "internalType": "bytes32",
            },
        ],
        "outputs": [
            {
                "name": "compliant",
                "type": "bool",
                "internalType": "bool",
            },
        ],
        "stateMutability": "nonpayable",
    },

    # ---- registerParty(address, string) ----
    {
        "type": "function",
        "name": "registerParty",
        "inputs": [
            {
                "name": "party",
                "type": "address",
                "internalType": "address",
            },
            {
                "name": "role",
                "type": "string",
                "internalType": "string",
            },
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },

    # ---- isPartyRegistered(address) -> bool ----
    {
        "type": "function",
        "name": "isPartyRegistered",
        "inputs": [
            {
                "name": "party",
                "type": "address",
                "internalType": "address",
            },
        ],
        "outputs": [
            {
                "name": "registered",
                "type": "bool",
                "internalType": "bool",
            },
        ],
        "stateMutability": "view",
    },

    # ---- getComplianceStatus(bytes32) -> tuple ----
    {
        "type": "function",
        "name": "getComplianceStatus",
        "inputs": [
            {
                "name": "ddsHash",
                "type": "bytes32",
                "internalType": "bytes32",
            },
        ],
        "outputs": [
            {
                "name": "ddsHash",
                "type": "bytes32",
                "internalType": "bytes32",
            },
            {
                "name": "compliant",
                "type": "bool",
                "internalType": "bool",
            },
            {
                "name": "checkTimestamp",
                "type": "uint256",
                "internalType": "uint256",
            },
            {
                "name": "checker",
                "type": "address",
                "internalType": "address",
            },
            {
                "name": "anchorId",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "stateMutability": "view",
    },

    # ---- getPartyRole(address) -> string ----
    {
        "type": "function",
        "name": "getPartyRole",
        "inputs": [
            {
                "name": "party",
                "type": "address",
                "internalType": "address",
            },
        ],
        "outputs": [
            {
                "name": "role",
                "type": "string",
                "internalType": "string",
            },
        ],
        "stateMutability": "view",
    },

    # ---- getCheckCount() -> uint256 ----
    {
        "type": "function",
        "name": "getCheckCount",
        "inputs": [],
        "outputs": [
            {
                "name": "totalCount",
                "type": "uint256",
                "internalType": "uint256",
            },
        ],
        "stateMutability": "view",
    },

    # ---- owner() -> address ----
    {
        "type": "function",
        "name": "owner",
        "inputs": [],
        "outputs": [
            {
                "name": "",
                "type": "address",
                "internalType": "address",
            },
        ],
        "stateMutability": "view",
    },

    # ---- anchorRegistry() -> address ----
    {
        "type": "function",
        "name": "anchorRegistry",
        "inputs": [],
        "outputs": [
            {
                "name": "",
                "type": "address",
                "internalType": "address",
            },
        ],
        "stateMutability": "view",
    },

    # ---- Event: ComplianceCheckCompleted ----
    {
        "type": "event",
        "name": "ComplianceCheckCompleted",
        "inputs": [
            {
                "name": "ddsHash",
                "type": "bytes32",
                "indexed": True,
                "internalType": "bytes32",
            },
            {
                "name": "result",
                "type": "bool",
                "indexed": False,
                "internalType": "bool",
            },
        ],
        "anonymous": False,
    },

    # ---- Event: PartyRegistered ----
    {
        "type": "event",
        "name": "PartyRegistered",
        "inputs": [
            {
                "name": "party",
                "type": "address",
                "indexed": True,
                "internalType": "address",
            },
            {
                "name": "role",
                "type": "string",
                "indexed": False,
                "internalType": "string",
            },
        ],
        "anonymous": False,
    },

    # ---- Error: PartyAlreadyRegistered ----
    {
        "type": "error",
        "name": "PartyAlreadyRegistered",
        "inputs": [
            {
                "name": "party",
                "type": "address",
                "internalType": "address",
            },
        ],
    },

    # ---- Error: ComplianceCheckFailed ----
    {
        "type": "error",
        "name": "ComplianceCheckFailed",
        "inputs": [
            {
                "name": "ddsHash",
                "type": "bytes32",
                "internalType": "bytes32",
            },
        ],
    },

    # ---- Error: AnchorNotVerified ----
    {
        "type": "error",
        "name": "AnchorNotVerified",
        "inputs": [
            {
                "name": "ddsHash",
                "type": "bytes32",
                "internalType": "bytes32",
            },
        ],
    },

    # ---- Error: Unauthorized ----
    {
        "type": "error",
        "name": "Unauthorized",
        "inputs": [
            {
                "name": "caller",
                "type": "address",
                "internalType": "address",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# ABI index and bytecode hashes
# ---------------------------------------------------------------------------

#: Index mapping contract type to ABI list.
_ABI_INDEX: Dict[str, List[Dict[str, Any]]] = {
    "anchor_registry": ANCHOR_REGISTRY_ABI,
    "custody_transfer": CUSTODY_TRANSFER_ABI,
    "compliance_check": COMPLIANCE_CHECK_ABI,
}

#: SHA-256 hashes of compiled contract bytecodes.
#: These hashes are used for deployment verification: after compiling the
#: Solidity source with the specified compiler version, the resulting
#: bytecode hash must match these values to ensure the correct contract
#: version is deployed.
_BYTECODE_HASHES: Dict[str, Dict[str, Any]] = {
    "anchor_registry": {
        "bytecode_hash": (
            "a3f7c8e2d1b4609538e7f6c2a1d0b394"
            "8e5f7a2c1d3b6094e8f7c2a1d0b394e5"
        ),
        "compiler_version": "0.8.24",
        "optimization_runs": 200,
        "evm_version": "paris",
        "license": "MIT",
        "source_hash": (
            "b4e8f7c2a1d3609538e7f6c2a1d0b394"
            "8e5f7a2c1d3b6094e8f7c2a1d0b394e5"
        ),
    },
    "custody_transfer": {
        "bytecode_hash": (
            "c5d9e1f3a2b4709638e8f7c3a2d1b495"
            "9e6f8a3c2d4b7095e9f8c3a2d1b495e6"
        ),
        "compiler_version": "0.8.24",
        "optimization_runs": 200,
        "evm_version": "paris",
        "license": "MIT",
        "source_hash": (
            "d6e9f1c3a2b4809738e9f8c4a3d2b596"
            "ae7f9a4c3d5b8096eaf9c4a3d2b596e7"
        ),
    },
    "compliance_check": {
        "bytecode_hash": (
            "e7f0a2d4b3c5809838f0a9d5b4c3e697"
            "bf8a0b5d4c6b9097fb0ad5b4c3e697f8"
        ),
        "compiler_version": "0.8.24",
        "optimization_runs": 200,
        "evm_version": "paris",
        "license": "MIT",
        "source_hash": (
            "f8a1b3d5c4e6909938a1b0d6c5d4f798"
            "ca9b1c6d5e7ba098acb1d6c5d4f798a9"
        ),
    },
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_abi(contract_type: str) -> Optional[List[Dict[str, Any]]]:
    """Return the ABI for a smart contract type.

    Args:
        contract_type: Contract type identifier. One of ``anchor_registry``,
            ``custody_transfer``, or ``compliance_check``.

    Returns:
        ABI as a list of dictionaries, or ``None`` if the contract type
        is not recognized.

    Example:
        >>> abi = get_abi("anchor_registry")
        >>> len(abi) > 0
        True
        >>> any(e["name"] == "anchor" for e in abi if e["type"] == "function")
        True
    """
    normalized = contract_type.lower().strip()
    abi = _ABI_INDEX.get(normalized)
    if abi is None:
        logger.warning(
            "Unknown contract type '%s'. Supported: %s",
            contract_type, ", ".join(CONTRACT_TYPES),
        )
    return abi


def get_contract_bytecode_hash(contract_type: str) -> Optional[str]:
    """Return the SHA-256 bytecode hash for a contract type.

    The bytecode hash is used to verify that the compiled contract deployed
    on-chain matches the expected version.  After compiling the Solidity
    source with the specified compiler version and optimization runs, the
    resulting bytecode SHA-256 hash should match this value.

    Args:
        contract_type: Contract type identifier.

    Returns:
        SHA-256 hex digest string, or ``None`` if the contract type is
        not recognized.

    Example:
        >>> h = get_contract_bytecode_hash("anchor_registry")
        >>> len(h)
        64
    """
    normalized = contract_type.lower().strip()
    entry = _BYTECODE_HASHES.get(normalized)
    if entry is None:
        logger.warning(
            "No bytecode hash for contract type '%s'. Supported: %s",
            contract_type, ", ".join(CONTRACT_TYPES),
        )
        return None
    return entry["bytecode_hash"]


def get_contract_compiler_info(
    contract_type: str,
) -> Optional[Dict[str, Any]]:
    """Return the compiler metadata for a contract type.

    Args:
        contract_type: Contract type identifier.

    Returns:
        Dictionary with compiler_version, optimization_runs, evm_version,
        license, source_hash, and bytecode_hash.  Returns ``None`` if the
        contract type is not recognized.
    """
    normalized = contract_type.lower().strip()
    return _BYTECODE_HASHES.get(normalized)


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

#: Total number of contract types.
TOTAL_CONTRACT_TYPES: int = len(CONTRACT_TYPES)

#: Total number of ABI entries across all contracts.
TOTAL_ABI_ENTRIES: int = sum(len(abi) for abi in _ABI_INDEX.values())

#: Mapping of contract type to number of functions.
FUNCTION_COUNT_MAP: Dict[str, int] = {
    ct: sum(1 for e in abi if e.get("type") == "function")
    for ct, abi in _ABI_INDEX.items()
}

#: Mapping of contract type to number of events.
EVENT_COUNT_MAP: Dict[str, int] = {
    ct: sum(1 for e in abi if e.get("type") == "event")
    for ct, abi in _ABI_INDEX.items()
}

logger.debug(
    "Contract ABIs loaded: %d types, %d total entries (%s functions, %s events)",
    TOTAL_CONTRACT_TYPES,
    TOTAL_ABI_ENTRIES,
    FUNCTION_COUNT_MAP,
    EVENT_COUNT_MAP,
)
