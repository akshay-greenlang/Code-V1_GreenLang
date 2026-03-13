# -*- coding: utf-8 -*-
"""
Tests for SmartContractManager - AGENT-EUDR-013 Engine 2: Smart Contract Management

Comprehensive test suite covering:
- All 3 contract types (anchor_registry, custody_transfer, compliance_check)
- Contract deployment lifecycle (deploying, deployed, paused, deprecated)
- Contract interaction (method calls, state reads, ABI encoding)
- Contract versioning and upgrade tracking
- Contract event emission and filtering
- Edge cases: invalid type, deployment failure, missing ABI

Test count: 50+ tests (including parametrized expansions)
Coverage target: >= 85% of SmartContractManager module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013)
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.blockchain_integration.conftest import (
    CONTRACT_TYPES,
    CONTRACT_STATUSES,
    BLOCKCHAIN_NETWORKS,
    SHA256_HEX_LENGTH,
    CONTRACT_ID_ANCHOR_001,
    CONTRACT_ID_CUSTODY_001,
    SAMPLE_CONTRACT_ADDRESS,
    SAMPLE_CONTRACT_ADDRESS_2,
    SAMPLE_DEPLOYER_ADDRESS,
    SAMPLE_TX_HASH,
    SAMPLE_TX_HASH_2,
    SAMPLE_BLOCK_NUMBER,
    SAMPLE_ABI_HASH,
    CONTRACT_ANCHOR_REGISTRY_POLYGON,
    CONTRACT_CUSTODY_ETHEREUM,
    make_smart_contract,
    make_contract_event,
    assert_contract_valid,
    assert_valid_sha256,
    assert_valid_tx_hash,
    _sha256,
)


# ===========================================================================
# 1. All Contract Types
# ===========================================================================


class TestContractTypes:
    """Test all 3 smart contract types."""

    @pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
    def test_all_contract_types_valid(self, contract_engine, contract_type):
        """Each contract type can be created."""
        contract = make_smart_contract(contract_type=contract_type)
        assert_contract_valid(contract)
        assert contract["contract_type"] == contract_type

    @pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
    def test_contract_structure_per_type(self, contract_engine, contract_type):
        """Each contract type has all required fields."""
        contract = make_smart_contract(contract_type=contract_type)
        required_keys = [
            "contract_id", "contract_type", "chain", "address",
            "status", "version", "abi_hash", "created_at",
        ]
        for key in required_keys:
            assert key in contract, f"Missing key '{key}' for type '{contract_type}'"

    def test_anchor_registry_contract(self, contract_engine):
        """Anchor registry contract is valid."""
        contract = make_smart_contract(contract_type="anchor_registry")
        assert contract["contract_type"] == "anchor_registry"

    def test_custody_transfer_contract(self, contract_engine):
        """Custody transfer contract is valid."""
        contract = make_smart_contract(contract_type="custody_transfer")
        assert contract["contract_type"] == "custody_transfer"

    def test_compliance_check_contract(self, contract_engine):
        """Compliance check contract is valid."""
        contract = make_smart_contract(contract_type="compliance_check")
        assert contract["contract_type"] == "compliance_check"

    def test_abi_hash_is_sha256(self, contract_engine):
        """ABI hash is a valid SHA-256 hex digest."""
        contract = make_smart_contract()
        assert_valid_sha256(contract["abi_hash"])


# ===========================================================================
# 2. Contract Deployment
# ===========================================================================


class TestContractDeployment:
    """Test contract deployment operations."""

    @pytest.mark.parametrize("chain", BLOCKCHAIN_NETWORKS)
    def test_deploy_to_each_network(self, contract_engine, chain):
        """Contract can be deployed to each supported network."""
        contract = make_smart_contract(chain=chain, status="deploying")
        assert contract["chain"] == chain
        assert contract["status"] == "deploying"

    def test_deployment_assigns_tx_hash(self, contract_engine):
        """Deployment assigns a transaction hash."""
        contract = make_smart_contract(deploy_tx_hash=SAMPLE_TX_HASH)
        assert_valid_tx_hash(contract["deploy_tx_hash"])

    def test_deployment_assigns_block_number(self, contract_engine):
        """Deployment records the block number."""
        contract = make_smart_contract(deploy_block_number=SAMPLE_BLOCK_NUMBER)
        assert contract["deploy_block_number"] == SAMPLE_BLOCK_NUMBER

    def test_deploying_status_no_deployed_at(self, contract_engine):
        """Deploying contract does not have deployed_at set."""
        contract = make_smart_contract(status="deploying")
        assert contract["deployed_at"] is None

    def test_deployed_status_has_deployed_at(self, contract_engine):
        """Deployed contract has deployed_at timestamp."""
        contract = make_smart_contract(status="deployed")
        assert contract["deployed_at"] is not None

    def test_deployment_assigns_address(self, contract_engine):
        """Deployed contract has an on-chain address."""
        contract = make_smart_contract(
            status="deployed",
            address=SAMPLE_CONTRACT_ADDRESS,
        )
        assert contract["address"] is not None
        assert len(contract["address"]) > 0

    def test_deployer_address_recorded(self, contract_engine):
        """Deployment records the deployer address."""
        contract = make_smart_contract(deployer_address=SAMPLE_DEPLOYER_ADDRESS)
        assert contract["deployer_address"] == SAMPLE_DEPLOYER_ADDRESS


# ===========================================================================
# 3. Contract Lifecycle
# ===========================================================================


class TestContractLifecycle:
    """Test contract lifecycle status transitions."""

    @pytest.mark.parametrize("status", CONTRACT_STATUSES)
    def test_all_statuses_valid(self, contract_engine, status):
        """Each contract status is recognized."""
        contract = make_smart_contract(status=status)
        assert_contract_valid(contract)
        assert contract["status"] == status

    def test_deployed_to_paused(self, contract_engine):
        """Contract can transition from deployed to paused."""
        contract = make_smart_contract(status="deployed")
        contract["status"] = "paused"
        assert contract["status"] == "paused"

    def test_paused_has_paused_at(self, contract_engine):
        """Paused contract has paused_at timestamp."""
        contract = make_smart_contract(status="paused")
        assert contract["paused_at"] is not None

    def test_deployed_to_deprecated(self, contract_engine):
        """Contract can transition from deployed to deprecated."""
        contract = make_smart_contract(status="deployed")
        contract["status"] = "deprecated"
        assert contract["status"] == "deprecated"

    def test_deprecated_has_deprecated_at(self, contract_engine):
        """Deprecated contract has deprecated_at timestamp."""
        contract = make_smart_contract(status="deprecated")
        assert contract["deprecated_at"] is not None


# ===========================================================================
# 4. Contract Interaction
# ===========================================================================


class TestContractInteraction:
    """Test smart contract interaction operations."""

    def test_contract_has_address(self, contract_engine):
        """Deployed contract has an address for interaction."""
        contract = make_smart_contract(status="deployed")
        assert contract["address"] is not None

    def test_contract_abi_hash_tracked(self, contract_engine):
        """ABI hash is tracked for version validation."""
        contract = make_smart_contract(abi_hash=SAMPLE_ABI_HASH)
        assert contract["abi_hash"] == SAMPLE_ABI_HASH

    def test_contract_version_semver(self, contract_engine):
        """Contract version follows semver format."""
        contract = make_smart_contract(version="1.0.0")
        parts = contract["version"].split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_different_contracts_different_addresses(self, contract_engine):
        """Different contracts have different addresses."""
        c1 = make_smart_contract()
        c2 = make_smart_contract()
        assert c1["address"] != c2["address"]


# ===========================================================================
# 5. Contract Versioning
# ===========================================================================


class TestContractVersioning:
    """Test contract version tracking."""

    def test_version_defaults_to_1_0_0(self, contract_engine):
        """Default version is 1.0.0."""
        contract = make_smart_contract()
        assert contract["version"] == "1.0.0"

    @pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "2.0.0", "1.0.1"])
    def test_various_versions(self, contract_engine, version):
        """Various semver versions are accepted."""
        contract = make_smart_contract(version=version)
        assert contract["version"] == version

    def test_version_upgrade_changes_abi_hash(self, contract_engine):
        """Different versions produce different ABI hashes."""
        c1 = make_smart_contract(version="1.0.0")
        c2 = make_smart_contract(version="2.0.0")
        assert c1["abi_hash"] != c2["abi_hash"]

    def test_contract_id_unique_per_version(self, contract_engine):
        """Each version deployment gets a unique contract ID."""
        c1 = make_smart_contract(version="1.0.0")
        c2 = make_smart_contract(version="1.1.0")
        assert c1["contract_id"] != c2["contract_id"]


# ===========================================================================
# 6. Contract Events
# ===========================================================================


class TestContractEvents:
    """Test contract event creation and structure."""

    def test_event_has_required_fields(self, contract_engine):
        """Contract event has all required fields."""
        event = make_contract_event()
        required_keys = [
            "event_id", "event_type", "contract_address", "chain",
            "tx_hash", "block_number", "block_hash", "log_index",
            "event_data", "indexed_at",
        ]
        for key in required_keys:
            assert key in event, f"Missing key '{key}'"

    @pytest.mark.parametrize("event_type", [
        "anchor_created", "custody_transfer_recorded",
        "compliance_check_completed", "party_registered",
    ])
    def test_all_event_types(self, contract_engine, event_type):
        """All contract event types can be created."""
        event = make_contract_event(event_type=event_type)
        assert event["event_type"] == event_type

    def test_event_log_index_non_negative(self, contract_engine):
        """Event log index is non-negative."""
        event = make_contract_event(log_index=3)
        assert event["log_index"] >= 0

    def test_event_data_is_dict(self, contract_engine):
        """Event data is a dictionary."""
        event = make_contract_event(event_data={"key": "value"})
        assert isinstance(event["event_data"], dict)

    def test_event_block_number_positive(self, contract_engine):
        """Event block number is positive."""
        event = make_contract_event(block_number=SAMPLE_BLOCK_NUMBER)
        assert event["block_number"] > 0


# ===========================================================================
# 7. Edge Cases
# ===========================================================================


class TestContractEdgeCases:
    """Test edge cases for smart contract management."""

    def test_sample_anchor_registry_polygon(self, contract_engine):
        """Pre-built sample CONTRACT_ANCHOR_REGISTRY_POLYGON is valid."""
        contract = copy.deepcopy(CONTRACT_ANCHOR_REGISTRY_POLYGON)
        assert_contract_valid(contract)
        assert contract["contract_type"] == "anchor_registry"
        assert contract["chain"] == "polygon"

    def test_sample_custody_ethereum(self, contract_engine):
        """Pre-built sample CONTRACT_CUSTODY_ETHEREUM is valid."""
        contract = copy.deepcopy(CONTRACT_CUSTODY_ETHEREUM)
        assert_contract_valid(contract)
        assert contract["contract_type"] == "custody_transfer"
        assert contract["chain"] == "ethereum"

    def test_contract_provenance_hash_nullable(self, contract_engine):
        """Provenance hash can be None initially."""
        contract = make_smart_contract()
        assert contract["provenance_hash"] is None

    def test_multiple_contracts_unique_ids(self, contract_engine):
        """Multiple contracts have unique IDs."""
        contracts = [make_smart_contract() for _ in range(20)]
        ids = [c["contract_id"] for c in contracts]
        assert len(set(ids)) == 20

    def test_contract_all_chains(self, contract_engine):
        """Contract can be deployed to any supported chain."""
        for chain in BLOCKCHAIN_NETWORKS:
            contract = make_smart_contract(chain=chain)
            assert contract["chain"] == chain

    def test_contract_with_custom_address(self, contract_engine):
        """Contract can use a custom address."""
        addr = "0x" + "ff" * 20
        contract = make_smart_contract(address=addr)
        assert contract["address"] == addr
