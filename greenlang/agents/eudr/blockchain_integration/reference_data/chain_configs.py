# -*- coding: utf-8 -*-
"""
Chain Configurations Reference Data - AGENT-EUDR-013 Blockchain Integration

Network configurations for all supported blockchain networks used in EUDR
compliance anchoring.  Each configuration entry provides chain identity,
RPC connectivity patterns, consensus parameters, block timing, confirmation
depth requirements, gas currency, and EIP-1559 support flags.

Supported Networks (6 configurations across 4 network families):
    - Ethereum Mainnet (chain_id=1): Public permissionless, 12s block time,
      12-block confirmation depth, EIP-1559 gas pricing.
    - Ethereum Goerli Testnet (chain_id=5): Public testnet for development
      and staging deployments.  Same confirmation depth as mainnet.
    - Polygon Mainnet (chain_id=137): EVM-compatible L2 with 2s block time,
      32-block confirmation depth, lower gas costs.  Primary chain for
      production EUDR anchoring.
    - Polygon Mumbai Testnet (chain_id=80001): Polygon testnet for
      development and integration testing.
    - Hyperledger Fabric: Permissioned enterprise network with channel-based
      privacy, orderer/peer architecture, MSP identity management, and
      single-block finality.
    - Hyperledger Besu: EVM-compatible enterprise chain with IBFT 2.0 / QBFT
      consensus, permissioning, single-block confirmation.

Network Categories:
    - EVM_NETWORKS: Ethereum, Polygon, Besu (share EVM execution model)
    - PERMISSIONED_NETWORKS: Fabric, Besu (enterprise permissioned chains)

Lookup Helpers:
    get_chain_config(network) -> dict | None
    get_default_rpc(network, environment) -> str
    get_confirmation_depth(network) -> int
    get_block_time_seconds(network) -> float

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013) - Appendix B
Agent ID: GL-EUDR-BCI-013
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported network identifiers
# ---------------------------------------------------------------------------

#: All supported blockchain network identifiers.
SUPPORTED_NETWORKS: List[str] = [
    "ethereum",
    "ethereum_goerli",
    "polygon",
    "polygon_mumbai",
    "fabric",
    "besu",
]

#: EVM-compatible networks (share Ethereum Virtual Machine execution model).
EVM_NETWORKS: List[str] = [
    "ethereum",
    "ethereum_goerli",
    "polygon",
    "polygon_mumbai",
    "besu",
]

#: Permissioned enterprise networks.
PERMISSIONED_NETWORKS: List[str] = [
    "fabric",
    "besu",
]

# ---------------------------------------------------------------------------
# Chain configurations: Dict[network_id, config_dict]
# ---------------------------------------------------------------------------

CHAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ==================================================================
    # Ethereum Mainnet
    # ==================================================================
    "ethereum": {
        "network_id": "ethereum",
        "display_name": "Ethereum Mainnet",
        "chain_id": 1,
        "network_type": "public",
        "consensus": "proof_of_stake",
        "evm_compatible": True,
        "block_time_seconds": 12.0,
        "confirmation_depth": 12,
        "gas_currency": "ETH",
        "gas_currency_decimals": 18,
        "eip1559_supported": True,
        "max_priority_fee_gwei": 2.0,
        "max_fee_gwei": 100.0,
        "rpc_url_patterns": {
            "development": "http://localhost:8545",
            "staging": "https://eth-goerli.g.alchemy.com/v2/{api_key}",
            "production": "https://eth-mainnet.g.alchemy.com/v2/{api_key}",
        },
        "rpc_timeout_s": 30,
        "max_connections": 10,
        "explorer_url": "https://etherscan.io",
        "explorer_api_url": "https://api.etherscan.io/api",
        "native_token": {
            "symbol": "ETH",
            "name": "Ether",
            "decimals": 18,
        },
        "contract_deploy_gas_limit": 3_000_000,
        "anchor_gas_limit": 150_000,
        "supported_contract_types": [
            "anchor_registry",
            "custody_transfer",
            "compliance_check",
        ],
        "eudr_notes": (
            "Ethereum Mainnet provides the highest decentralization and "
            "censorship resistance for EUDR compliance anchoring.  Higher "
            "gas costs compared to Polygon but stronger immutability "
            "guarantees.  Recommended for high-value DDS submissions "
            "and competent authority audit anchors."
        ),
    },

    # ==================================================================
    # Ethereum Goerli Testnet
    # ==================================================================
    "ethereum_goerli": {
        "network_id": "ethereum_goerli",
        "display_name": "Ethereum Goerli Testnet",
        "chain_id": 5,
        "network_type": "testnet",
        "consensus": "proof_of_authority",
        "evm_compatible": True,
        "block_time_seconds": 12.0,
        "confirmation_depth": 12,
        "gas_currency": "GoerliETH",
        "gas_currency_decimals": 18,
        "eip1559_supported": True,
        "max_priority_fee_gwei": 1.0,
        "max_fee_gwei": 50.0,
        "rpc_url_patterns": {
            "development": "http://localhost:8545",
            "staging": "https://eth-goerli.g.alchemy.com/v2/{api_key}",
            "production": "https://eth-goerli.g.alchemy.com/v2/{api_key}",
        },
        "rpc_timeout_s": 30,
        "max_connections": 5,
        "explorer_url": "https://goerli.etherscan.io",
        "explorer_api_url": "https://api-goerli.etherscan.io/api",
        "native_token": {
            "symbol": "GoerliETH",
            "name": "Goerli Ether",
            "decimals": 18,
        },
        "contract_deploy_gas_limit": 3_000_000,
        "anchor_gas_limit": 150_000,
        "supported_contract_types": [
            "anchor_registry",
            "custody_transfer",
            "compliance_check",
        ],
        "eudr_notes": (
            "Goerli testnet is used for development and staging environments. "
            "Free test ETH available from faucets.  Same EVM execution as "
            "mainnet for contract compatibility testing."
        ),
    },

    # ==================================================================
    # Polygon Mainnet
    # ==================================================================
    "polygon": {
        "network_id": "polygon",
        "display_name": "Polygon PoS Mainnet",
        "chain_id": 137,
        "network_type": "public",
        "consensus": "proof_of_stake",
        "evm_compatible": True,
        "block_time_seconds": 2.0,
        "confirmation_depth": 32,
        "gas_currency": "MATIC",
        "gas_currency_decimals": 18,
        "eip1559_supported": True,
        "max_priority_fee_gwei": 30.0,
        "max_fee_gwei": 300.0,
        "rpc_url_patterns": {
            "development": "http://localhost:8546",
            "staging": "https://polygon-mumbai.g.alchemy.com/v2/{api_key}",
            "production": "https://polygon-mainnet.g.alchemy.com/v2/{api_key}",
        },
        "rpc_timeout_s": 15,
        "max_connections": 15,
        "explorer_url": "https://polygonscan.com",
        "explorer_api_url": "https://api.polygonscan.com/api",
        "native_token": {
            "symbol": "MATIC",
            "name": "Polygon MATIC",
            "decimals": 18,
        },
        "contract_deploy_gas_limit": 5_000_000,
        "anchor_gas_limit": 200_000,
        "supported_contract_types": [
            "anchor_registry",
            "custody_transfer",
            "compliance_check",
        ],
        "eudr_notes": (
            "Polygon PoS is the PRIMARY production chain for EUDR compliance "
            "anchoring due to low gas costs (~0.01 USD per anchor), fast "
            "confirmation (~64s at 32 blocks), and EVM compatibility.  "
            "Checkpoints are periodically committed to Ethereum L1 for "
            "additional security."
        ),
    },

    # ==================================================================
    # Polygon Mumbai Testnet
    # ==================================================================
    "polygon_mumbai": {
        "network_id": "polygon_mumbai",
        "display_name": "Polygon Mumbai Testnet",
        "chain_id": 80001,
        "network_type": "testnet",
        "consensus": "proof_of_stake",
        "evm_compatible": True,
        "block_time_seconds": 2.0,
        "confirmation_depth": 32,
        "gas_currency": "MumbaiMATIC",
        "gas_currency_decimals": 18,
        "eip1559_supported": True,
        "max_priority_fee_gwei": 30.0,
        "max_fee_gwei": 300.0,
        "rpc_url_patterns": {
            "development": "http://localhost:8546",
            "staging": "https://polygon-mumbai.g.alchemy.com/v2/{api_key}",
            "production": "https://polygon-mumbai.g.alchemy.com/v2/{api_key}",
        },
        "rpc_timeout_s": 15,
        "max_connections": 5,
        "explorer_url": "https://mumbai.polygonscan.com",
        "explorer_api_url": "https://api-testnet.polygonscan.com/api",
        "native_token": {
            "symbol": "MumbaiMATIC",
            "name": "Mumbai Test MATIC",
            "decimals": 18,
        },
        "contract_deploy_gas_limit": 5_000_000,
        "anchor_gas_limit": 200_000,
        "supported_contract_types": [
            "anchor_registry",
            "custody_transfer",
            "compliance_check",
        ],
        "eudr_notes": (
            "Mumbai testnet mirrors Polygon PoS for development and staging. "
            "Free test MATIC available from faucets.  Use for integration "
            "testing before production deployment."
        ),
    },

    # ==================================================================
    # Hyperledger Fabric
    # ==================================================================
    "fabric": {
        "network_id": "fabric",
        "display_name": "Hyperledger Fabric",
        "chain_id": None,
        "network_type": "permissioned",
        "consensus": "raft",
        "evm_compatible": False,
        "block_time_seconds": 2.0,
        "confirmation_depth": 1,
        "gas_currency": None,
        "gas_currency_decimals": 0,
        "eip1559_supported": False,
        "max_priority_fee_gwei": 0.0,
        "max_fee_gwei": 0.0,
        "rpc_url_patterns": {
            "development": "grpc://localhost:7051",
            "staging": "grpcs://fabric-peer.staging.greenlang.io:7051",
            "production": "grpcs://fabric-peer.prod.greenlang.io:7051",
        },
        "rpc_timeout_s": 10,
        "max_connections": 5,
        "explorer_url": None,
        "explorer_api_url": None,
        "native_token": None,
        "contract_deploy_gas_limit": 0,
        "anchor_gas_limit": 0,
        "supported_contract_types": [
            "anchor_registry",
            "custody_transfer",
        ],
        "channel_config": {
            "channel_name": "eudr-supply-chain",
            "chaincode_name": "eudr-anchor",
            "chaincode_version": "1.0.0",
            "endorsement_policy": "AND('Org1MSP.peer', 'Org2MSP.peer')",
            "collection_config": {
                "name": "eudr-private-data",
                "policy": "OR('Org1MSP.member', 'Org2MSP.member')",
                "required_peer_count": 1,
                "max_peer_count": 3,
                "block_to_live": 0,
                "member_only_read": True,
                "member_only_write": True,
            },
        },
        "orderer_config": {
            "orderer_url": "grpcs://orderer.greenlang.io:7050",
            "orderer_tls_enabled": True,
            "orderer_timeout_s": 30,
            "batch_timeout": "2s",
            "max_message_count": 500,
            "absolute_max_bytes": 10_485_760,
            "preferred_max_bytes": 2_097_152,
        },
        "peer_config": {
            "peer_urls": [
                "grpcs://peer0.org1.greenlang.io:7051",
                "grpcs://peer1.org1.greenlang.io:7051",
            ],
            "peer_tls_enabled": True,
            "gossip_bootstrap": "peer0.org1.greenlang.io:7051",
            "gossip_use_leader_election": True,
            "gossip_org_leader": False,
        },
        "msp_config": {
            "msp_id": "Org1MSP",
            "crypto_path": "/etc/hyperledger/fabric/msp",
            "admin_certs_path": "/etc/hyperledger/fabric/msp/admincerts",
            "ca_certs_path": "/etc/hyperledger/fabric/msp/cacerts",
            "keystore_path": "/etc/hyperledger/fabric/msp/keystore",
            "signcerts_path": "/etc/hyperledger/fabric/msp/signcerts",
            "tls_root_certs": "/etc/hyperledger/fabric/tls/ca.crt",
        },
        "eudr_notes": (
            "Hyperledger Fabric provides channel-level data isolation for "
            "confidential EUDR supply chain data.  Single-block finality "
            "eliminates confirmation waiting.  Private data collections "
            "support selective disclosure to competent authorities per "
            "Article 14.  Recommended for consortium deployments where "
            "all parties are known."
        ),
    },

    # ==================================================================
    # Hyperledger Besu
    # ==================================================================
    "besu": {
        "network_id": "besu",
        "display_name": "Hyperledger Besu",
        "chain_id": 1337,
        "network_type": "permissioned",
        "consensus": "ibft2",
        "evm_compatible": True,
        "block_time_seconds": 5.0,
        "confirmation_depth": 1,
        "gas_currency": "BesuETH",
        "gas_currency_decimals": 18,
        "eip1559_supported": False,
        "max_priority_fee_gwei": 0.0,
        "max_fee_gwei": 0.0,
        "rpc_url_patterns": {
            "development": "http://localhost:8547",
            "staging": "https://besu.staging.greenlang.io:8545",
            "production": "https://besu.prod.greenlang.io:8545",
        },
        "rpc_timeout_s": 15,
        "max_connections": 10,
        "explorer_url": None,
        "explorer_api_url": None,
        "native_token": {
            "symbol": "BesuETH",
            "name": "Besu Ether",
            "decimals": 18,
        },
        "contract_deploy_gas_limit": 5_000_000,
        "anchor_gas_limit": 200_000,
        "supported_contract_types": [
            "anchor_registry",
            "custody_transfer",
            "compliance_check",
        ],
        "consensus_config": {
            "algorithm": "ibft2",
            "block_period_seconds": 5,
            "epoch_length": 30_000,
            "request_timeout_seconds": 10,
            "validators": [
                "0x0000000000000000000000000000000000000001",
                "0x0000000000000000000000000000000000000002",
                "0x0000000000000000000000000000000000000003",
                "0x0000000000000000000000000000000000000004",
            ],
            "min_validators": 4,
            "max_validators": 20,
            "duplicate_message_limit": 100,
            "future_messages_limit": 1000,
            "future_messages_max_distance": 10,
        },
        "alternative_consensus": {
            "algorithm": "qbft",
            "block_period_seconds": 5,
            "epoch_length": 30_000,
            "request_timeout_seconds": 10,
        },
        "permissioning_config": {
            "enabled": True,
            "accounts_allowlist_enabled": True,
            "nodes_allowlist_enabled": True,
            "accounts_contract_enabled": False,
            "nodes_contract_enabled": False,
            "accounts_allowlist": [],
            "nodes_allowlist": [],
        },
        "privacy_config": {
            "enabled": True,
            "privacy_url": "http://localhost:8888",
            "privacy_precompiled_address": "0x000000000000000000000000000000000000007e",
            "privacy_marker_transaction_signing_key": "",
            "privacy_public_key_file": "/etc/besu/privacy/key.pub",
            "privacy_flexible_groups_enabled": True,
        },
        "eudr_notes": (
            "Hyperledger Besu provides EVM compatibility with enterprise "
            "permissioning for consortium EUDR deployments.  IBFT 2.0 / QBFT "
            "consensus provides Byzantine fault tolerance with single-block "
            "finality.  Account and node permissioning restricts network "
            "access to authorized supply chain participants.  Privacy groups "
            "support selective data sharing per Article 14."
        ),
    },
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_chain_config(network: str) -> Optional[Dict[str, Any]]:
    """Return the full configuration for a blockchain network.

    Args:
        network: Network identifier (e.g. ``"ethereum"``, ``"polygon"``).
            For Ethereum mainnet, both ``"ethereum"`` and ``"ethereum_mainnet"``
            are accepted.

    Returns:
        Configuration dictionary or ``None`` if the network is not found.

    Example:
        >>> cfg = get_chain_config("polygon")
        >>> cfg["chain_id"]
        137
    """
    # Normalize common aliases
    normalized = network.lower().strip()
    aliases: Dict[str, str] = {
        "ethereum_mainnet": "ethereum",
        "eth": "ethereum",
        "eth_mainnet": "ethereum",
        "eth_goerli": "ethereum_goerli",
        "goerli": "ethereum_goerli",
        "polygon_mainnet": "polygon",
        "polygon_pos": "polygon",
        "matic": "polygon",
        "mumbai": "polygon_mumbai",
        "hyperledger_fabric": "fabric",
        "hlf": "fabric",
        "hyperledger_besu": "besu",
    }
    resolved = aliases.get(normalized, normalized)
    config = CHAIN_CONFIGS.get(resolved)
    if config is None:
        logger.warning(
            "Unknown blockchain network '%s' (resolved='%s'). "
            "Supported: %s",
            network, resolved, ", ".join(SUPPORTED_NETWORKS),
        )
    return config


def get_default_rpc(
    network: str,
    environment: str = "development",
) -> str:
    """Return the default RPC URL for a network and environment.

    Args:
        network: Network identifier.
        environment: Deployment environment (``development``, ``staging``,
            or ``production``).

    Returns:
        RPC URL string.  Returns ``"http://localhost:8545"`` as fallback
        if the network or environment is not found.

    Example:
        >>> get_default_rpc("polygon", "production")
        'https://polygon-mainnet.g.alchemy.com/v2/{api_key}'
    """
    config = get_chain_config(network)
    if config is None:
        logger.warning(
            "No RPC URL for unknown network '%s'; using localhost fallback",
            network,
        )
        return "http://localhost:8545"

    rpc_patterns = config.get("rpc_url_patterns", {})
    env_normalized = environment.lower().strip()
    url = rpc_patterns.get(env_normalized)
    if url is None:
        # Fall back to development URL
        url = rpc_patterns.get("development", "http://localhost:8545")
        logger.debug(
            "No RPC URL for environment '%s' on '%s'; "
            "falling back to development URL",
            environment, network,
        )
    return url


def get_confirmation_depth(network: str) -> int:
    """Return the required confirmation depth for a network.

    The confirmation depth is the number of blocks that must be mined
    after a transaction's block before the transaction is considered
    final and irreversible.

    Args:
        network: Network identifier.

    Returns:
        Confirmation depth as an integer.  Returns ``12`` (Ethereum
        default) if the network is not found.

    Example:
        >>> get_confirmation_depth("polygon")
        32
        >>> get_confirmation_depth("fabric")
        1
    """
    config = get_chain_config(network)
    if config is None:
        return 12  # Default to Ethereum depth
    return config.get("confirmation_depth", 12)


def get_block_time_seconds(network: str) -> float:
    """Return the average block time in seconds for a network.

    Args:
        network: Network identifier.

    Returns:
        Block time in seconds as a float.  Returns ``12.0`` (Ethereum
        default) if the network is not found.

    Example:
        >>> get_block_time_seconds("polygon")
        2.0
        >>> get_block_time_seconds("besu")
        5.0
    """
    config = get_chain_config(network)
    if config is None:
        return 12.0  # Default to Ethereum block time
    return config.get("block_time_seconds", 12.0)


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

#: Total number of supported networks.
TOTAL_NETWORKS: int = len(SUPPORTED_NETWORKS)

#: Total number of EVM-compatible networks.
TOTAL_EVM_NETWORKS: int = len(EVM_NETWORKS)

#: Total number of permissioned networks.
TOTAL_PERMISSIONED_NETWORKS: int = len(PERMISSIONED_NETWORKS)

#: Mapping of network to chain_id (EVM networks only).
CHAIN_ID_MAP: Dict[str, Optional[int]] = {
    net: CHAIN_CONFIGS[net].get("chain_id")
    for net in SUPPORTED_NETWORKS
    if net in CHAIN_CONFIGS
}

#: Mapping of network to confirmation depth.
CONFIRMATION_DEPTH_MAP: Dict[str, int] = {
    net: CHAIN_CONFIGS[net].get("confirmation_depth", 12)
    for net in SUPPORTED_NETWORKS
    if net in CHAIN_CONFIGS
}

#: Mapping of network to block time in seconds.
BLOCK_TIME_MAP: Dict[str, float] = {
    net: CHAIN_CONFIGS[net].get("block_time_seconds", 12.0)
    for net in SUPPORTED_NETWORKS
    if net in CHAIN_CONFIGS
}

logger.debug(
    "Chain configs loaded: %d networks (%d EVM, %d permissioned)",
    TOTAL_NETWORKS,
    TOTAL_EVM_NETWORKS,
    TOTAL_PERMISSIONED_NETWORKS,
)
