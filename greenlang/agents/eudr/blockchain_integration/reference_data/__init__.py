# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-013: Blockchain Integration Agent

Provides built-in reference datasets for blockchain integration operations:
    - chain_configs: Network configurations for Ethereum, Polygon,
      Hyperledger Fabric, and Hyperledger Besu including chain IDs,
      RPC URL patterns, block times, confirmation depths, gas currencies,
      consensus parameters, and permissioning configuration.
    - contract_abis: Solidity ABI definitions for AnchorRegistry,
      CustodyTransfer, and ComplianceCheck smart contracts deployed
      for EUDR compliance anchoring, including function signatures,
      event definitions, and bytecode hashes.
    - anchor_rules: Anchoring rules per EUDR supply chain event type
      covering priority levels (P0/P1/P2), batch eligibility,
      required fields, validation rules, gas estimates per network,
      and EUDR Article 14 five-year retention policy.

These datasets enable deterministic, zero-hallucination blockchain
integration without hardcoded values scattered across engine modules.
All data is version-tracked and provenance-auditable per EU 2023/1115
Article 14 record-keeping requirements.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013)
Agent ID: GL-EUDR-BCI-013
"""

from greenlang.agents.eudr.blockchain_integration.reference_data.chain_configs import (
    CHAIN_CONFIGS,
    EVM_NETWORKS,
    PERMISSIONED_NETWORKS,
    SUPPORTED_NETWORKS,
    get_block_time_seconds,
    get_chain_config,
    get_confirmation_depth,
    get_default_rpc,
)
from greenlang.agents.eudr.blockchain_integration.reference_data.contract_abis import (
    ANCHOR_REGISTRY_ABI,
    COMPLIANCE_CHECK_ABI,
    CONTRACT_TYPES,
    CUSTODY_TRANSFER_ABI,
    get_abi,
    get_contract_bytecode_hash,
)
from greenlang.agents.eudr.blockchain_integration.reference_data.anchor_rules import (
    ANCHOR_RULES,
    GAS_ESTIMATES,
    RETENTION_RULES,
    get_anchor_rule,
    get_max_batch_wait,
    get_priority,
    get_required_fields,
    is_batch_eligible,
    validate_anchor_request,
)

__all__ = [
    # -- chain_configs --
    "CHAIN_CONFIGS",
    "SUPPORTED_NETWORKS",
    "EVM_NETWORKS",
    "PERMISSIONED_NETWORKS",
    "get_chain_config",
    "get_default_rpc",
    "get_confirmation_depth",
    "get_block_time_seconds",
    # -- contract_abis --
    "ANCHOR_REGISTRY_ABI",
    "CUSTODY_TRANSFER_ABI",
    "COMPLIANCE_CHECK_ABI",
    "CONTRACT_TYPES",
    "get_abi",
    "get_contract_bytecode_hash",
    # -- anchor_rules --
    "ANCHOR_RULES",
    "GAS_ESTIMATES",
    "RETENTION_RULES",
    "get_anchor_rule",
    "get_priority",
    "is_batch_eligible",
    "get_max_batch_wait",
    "get_required_fields",
    "validate_anchor_request",
]
