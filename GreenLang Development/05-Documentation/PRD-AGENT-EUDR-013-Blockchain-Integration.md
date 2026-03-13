# PRD: AGENT-EUDR-013 -- Blockchain Integration Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-013 |
| **Agent ID** | GL-EUDR-BCI-013 |
| **Component** | Blockchain Integration Agent |
| **Category** | EUDR Regulatory Agent -- Immutable Traceability & Distributed Ledger |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-08 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-08 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation mandates that operators retain traceability records for a minimum of 5 years (Article 14) and that competent authorities can verify the provenance of every commodity placed on the EU market. Traditional database-backed traceability systems are vulnerable to several critical weaknesses that undermine regulatory trust:

- **Data mutability**: Centralized databases allow administrators to modify, delete, or backdate records without leaving a detectable audit trail. A compliance officer could alter a shipment record to show a different country of origin months after the original entry, and no external party could detect the change.
- **Single point of trust**: Regulators must trust the operator's own system to provide accurate historical data. When an operator is under investigation for EUDR non-compliance, the very records under scrutiny are stored in systems controlled by the investigated party.
- **Cross-party data disputes**: When a supplier claims they delivered 1,000 tonnes of compliant cocoa but the importer's system shows only 800 tonnes, there is no neutral, tamper-proof record to resolve the dispute. Each party can modify their own records, making reconciliation impossible.
- **Audit trail integrity**: Internal audit logs can be modified by database administrators. SHA-256 provenance hashes (as implemented in agents 001-012) provide tamper detection but not tamper prevention -- a sufficiently motivated actor can recompute the entire hash chain after modifying historical records.
- **Multi-stakeholder visibility**: EUDR supply chains involve producers, collectors, processors, traders, importers, certification bodies, and competent authorities. No single party's database is trusted by all others. Shared visibility requires a neutral, append-only ledger.
- **Certification body verification**: When FSC or RSPO certificates are referenced in a DDS, competent authorities cannot independently verify that the certificate was valid at the time of the transaction without querying the certification body's database, which may have been updated since.
- **Temporal proof**: Regulators need proof that a specific record existed at a specific time. Traditional timestamps from application servers can be falsified. Blockchain timestamps provide cryptographic proof of existence tied to block creation time.
- **Supply chain event ordering**: When custody transfers happen across multiple time zones and systems, establishing the definitive chronological order of events is challenging. Blockchain provides a canonical, globally agreed ordering of transactions.

Without distributed ledger integration, EU operators cannot provide the level of data integrity, transparency, and non-repudiation that competent authorities increasingly expect, particularly for high-risk supply chains where EUDR penalties can reach 4% of annual EU turnover.

### 1.2 Solution Overview

Agent-EUDR-013: Blockchain Integration Agent provides a production-grade, blockchain-agnostic integration layer that anchors EUDR traceability records to distributed ledgers for immutable proof of existence, tamper-proof audit trails, and multi-stakeholder data sharing. It supports multiple blockchain protocols while maintaining the zero-hallucination principle through deterministic on-chain operations.

Core capabilities:

1. **Transaction anchoring engine** -- Anchors SHA-256 hashes of critical EUDR records (DDS submissions, custody transfers, batch events, certification references, reconciliation results) to blockchain networks, creating immutable proof of existence with cryptographic timestamps.
2. **Smart contract manager** -- Deploys, manages, and interacts with EUDR-specific smart contracts that enforce compliance rules on-chain: custody transfer validation, mass balance verification, certificate validity checking, and deforestation cutoff date enforcement.
3. **Multi-chain connector** -- Connects to multiple blockchain protocols (Ethereum/Polygon, Hyperledger Fabric, Hyperledger Besu) through a unified abstraction layer, enabling operators to choose their preferred network while maintaining interoperability.
4. **On-chain verification engine** -- Verifies that anchored records have not been tampered with by comparing local record hashes against on-chain anchored hashes, providing independent verification without trusting the operator's database.
5. **Event listener and indexer** -- Listens for on-chain events (new anchors, contract state changes, cross-party confirmations) and indexes them for fast off-chain querying, maintaining a synchronized view of blockchain state.
6. **Merkle proof generator** -- Generates compact Merkle inclusion proofs that allow any party to independently verify that a specific record was included in a batch anchor, without downloading the entire dataset.
7. **Cross-party data sharing** -- Enables secure, permissioned data sharing between supply chain parties through on-chain access control, allowing importers to share traceability data with competent authorities while protecting commercially sensitive information.
8. **Compliance evidence packager** -- Generates blockchain-backed compliance evidence packages that combine on-chain anchors with off-chain data for regulatory submissions, providing cryptographically verifiable proof of compliance history.

### 1.3 Dependencies

| Dependency | Component | Integration |
|------------|-----------|-------------|
| AGENT-EUDR-001 | Supply Chain Mapping Master | Supply chain graph nodes/edges for custody transfer anchoring |
| AGENT-EUDR-009 | Chain of Custody Agent | CoC events for on-chain recording |
| AGENT-EUDR-011 | Mass Balance Calculator | Ledger entries and reconciliation results for anchoring |
| AGENT-EUDR-012 | Document Authentication | Document hashes for on-chain anchoring |
| AGENT-DATA-005 | EUDR Traceability Connector | DDS data for submission anchoring |
| SEC-005 | Centralized Audit Logging | Audit events for blockchain anchoring |

---

## 2. Regulatory Context

### 2.1 EUDR Articles Addressed

| Article | Requirement | Agent Feature |
|---------|-------------|---------------|
| Art. 4(2) | Due diligence information collection | Transaction anchoring provides immutable evidence of data collection |
| Art. 9(1) | Information requirements for DDS | Smart contracts enforce DDS completeness before anchoring |
| Art. 10(2)(a) | Supply chain complexity assessment | Cross-party data sharing enables multi-tier visibility |
| Art. 12 | DDS submission to EU Information System | DDS anchor creates immutable proof of submission content |
| Art. 14 | 5-year record retention | Blockchain provides permanent, tamper-proof record retention |
| Art. 16 | Risk mitigation measures | On-chain verification enables independent compliance checking |
| Art. 29 | Country benchmarking | Smart contracts enforce country risk rules at transaction level |
| Art. 31 | Review and reporting | Compliance evidence packages provide blockchain-backed reports |

### 2.2 Supported Blockchain Protocols

| Protocol | Type | Use Case | Gas/Cost Model |
|----------|------|----------|----------------|
| Ethereum Mainnet | Public | High-value anchors, maximum transparency | ETH gas fees |
| Polygon (PoS) | Public L2 | Cost-effective anchoring, fast finality | MATIC gas fees |
| Hyperledger Fabric | Private/Permissioned | Enterprise consortia, confidential transactions | No gas fees |
| Hyperledger Besu | Private/Public | EVM-compatible enterprise networks | Configurable |

### 2.3 Anchor Event Types

| Event Type | Description | Frequency | Priority |
|------------|-------------|-----------|----------|
| DDS Submission | Hash of DDS content anchored at submission time | Per DDS | P0 |
| Custody Transfer | Hash of custody transfer record between parties | Per transfer | P0 |
| Batch Event | Hash of batch creation, split, merge, or transformation | Per event | P0 |
| Certificate Reference | Hash of certificate validity snapshot at transaction time | Per reference | P1 |
| Reconciliation Result | Hash of period-end reconciliation outcome | Per period | P1 |
| Mass Balance Entry | Hash of mass balance ledger entry | Per entry | P2 |
| Document Authentication | Hash of document authentication result | Per document | P2 |
| Geolocation Verification | Hash of plot geolocation verification result | Per verification | P2 |

---

## 3. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Anchor latency (public chain) | < 60 seconds to confirmation | Block confirmation time |
| Anchor latency (private chain) | < 5 seconds to confirmation | Block confirmation time |
| Verification accuracy | 100% tamper detection | Comparison of on-chain vs off-chain hashes |
| Anchor throughput | >= 1,000 anchors/minute (batched) | Batch anchoring benchmark |
| Merkle proof size | < 1 KB per proof | Proof serialization size |
| Merkle proof verification | < 10ms per proof | Verification benchmark |
| Smart contract gas efficiency | < 100,000 gas per anchor batch | Gas usage measurement |
| Multi-chain support | >= 3 blockchain protocols | Protocol coverage |
| Cross-party verification | Independent verification without operator DB access | Verification test |
| Test coverage | >= 500 unit tests | Pytest count |

---

## 4. Scope

### 4.1 In Scope
- Transaction anchoring for all 8 EUDR event types
- Smart contract deployment and interaction for Ethereum/Polygon and Hyperledger
- Multi-chain abstraction layer with unified API
- On-chain verification of anchored records
- Merkle tree batching for cost-efficient anchoring
- Event listening and off-chain indexing
- Cross-party permissioned data sharing
- Blockchain-backed compliance evidence packages
- Gas cost estimation and optimization
- 5-year anchor retention per EUDR Article 14

### 4.2 Out of Scope
- Cryptocurrency token creation or management
- DeFi protocol integration
- NFT minting for commodity tokens
- On-chain data storage (only hashes are stored on-chain)
- Blockchain node operation (connects to existing nodes/services)
- Consensus mechanism design
- Cross-chain bridge protocols
- Real-time on-chain analytics dashboards

---

## 5. Zero-Hallucination Principles

1. All on-chain operations use deterministic cryptographic functions (SHA-256, keccak256, ECDSA) -- no LLM inference.
2. Hash anchoring is exact byte-level: SHA-256(record_bytes) is stored on-chain unchanged.
3. Merkle tree construction is deterministic: same inputs produce identical root hash.
4. Smart contract logic uses Solidity/Chaincode with deterministic execution -- no probabilistic operations.
5. Block confirmation is binary (confirmed/not confirmed) -- no probability-based finality estimation.
6. Gas cost estimation uses deterministic formulas based on transaction data size and network base fee.
7. Cross-party verification is pure hash comparison -- on-chain hash == off-chain hash means no tampering.
8. SHA-256 provenance hashing ensures tamper detection on all off-chain records before anchoring.

---

## 6. Feature Requirements

### 6.1 Feature 1: Transaction Anchoring Engine (P0)

**Requirements**:
- F1.1: Anchor SHA-256 hashes of EUDR records to supported blockchain networks
- F1.2: Support 8 anchor event types (DDS, custody transfer, batch, certificate, reconciliation, mass balance, document auth, geolocation)
- F1.3: Batch anchoring: aggregate multiple record hashes into a single Merkle root for cost-efficient on-chain storage
- F1.4: Configurable batch size (default 100 records) and batch interval (default 5 minutes)
- F1.5: Anchor receipt: return blockchain transaction hash, block number, timestamp, Merkle root, and individual record proof
- F1.6: Retry logic: automatic retry with exponential backoff for failed transactions (network issues, gas estimation errors)
- F1.7: Anchor status tracking: pending, submitted, confirmed, failed, expired
- F1.8: Priority anchoring: immediate single-record anchoring for high-priority events (P0 event types)
- F1.9: Anchor history: full history of all anchored records with blockchain references
- F1.10: Gas cost tracking: record actual gas cost per anchor transaction for cost reporting

### 6.2 Feature 2: Smart Contract Manager (P0)

**Requirements**:
- F2.1: Deploy EUDR compliance smart contracts to supported networks
- F2.2: Contract types: AnchorRegistry (stores Merkle roots), CustodyTransfer (validates transfers), ComplianceCheck (enforces rules)
- F2.3: Contract versioning: support multiple contract versions with migration path
- F2.4: Contract interaction: read/write operations with proper gas estimation
- F2.5: ABI management: store and manage contract ABIs for all deployed contracts
- F2.6: Contract event subscription: listen for on-chain events emitted by EUDR contracts
- F2.7: Contract state queries: read current contract state (total anchors, registered parties, compliance status)
- F2.8: Multi-network deployment: deploy same contract logic to multiple networks
- F2.9: Contract upgrade: support proxy pattern for upgradeable contracts
- F2.10: Gas estimation: estimate gas cost before submitting transactions

### 6.3 Feature 3: Multi-Chain Connector (P0)

**Requirements**:
- F3.1: Unified abstraction layer for interacting with multiple blockchain protocols
- F3.2: Ethereum/Polygon connector: Web3 JSON-RPC, EIP-1559 transaction support, event filtering
- F3.3: Hyperledger Fabric connector: gRPC gateway, channel management, chaincode invocation
- F3.4: Hyperledger Besu connector: EVM-compatible JSON-RPC, privacy groups, permissioning
- F3.5: Connection health monitoring: automatic reconnection on connection loss
- F3.6: Network selection: configurable primary and fallback networks
- F3.7: Transaction signing: secure key management for transaction signing (HSM-compatible interface)
- F3.8: Nonce management: automatic nonce tracking to prevent transaction conflicts
- F3.9: Block confirmation tracking: configurable confirmation depth (default 12 for Ethereum, 1 for Fabric)
- F3.10: Network fee estimation: real-time gas/fee estimation per network

### 6.4 Feature 4: On-Chain Verification Engine (P0)

**Requirements**:
- F4.1: Verify record integrity by comparing local SHA-256 hash against on-chain anchored hash
- F4.2: Merkle proof verification: verify individual record inclusion using Merkle proof path
- F4.3: Batch verification: verify multiple records against a single Merkle root
- F4.4: Cross-chain verification: verify anchors across multiple blockchain networks
- F4.5: Temporal verification: confirm that a record was anchored at or before a claimed timestamp
- F4.6: Independent verification: verification works with only the on-chain data and the record hash (no database access required)
- F4.7: Verification report: detailed verification result with blockchain reference, proof path, and tamper status
- F4.8: Bulk verification: verify large sets of records efficiently using batch Merkle proofs
- F4.9: Verification cache: cache recent verification results with configurable TTL
- F4.10: Third-party verification API: allow external parties (auditors, regulators) to verify records without platform access

### 6.5 Feature 5: Event Listener and Indexer (P0)

**Requirements**:
- F5.1: Listen for on-chain events from deployed EUDR smart contracts
- F5.2: Event types: AnchorCreated, CustodyTransferRecorded, ComplianceCheckCompleted, PartyRegistered
- F5.3: Real-time event processing: process events within 1 block of emission
- F5.4: Event indexing: index events by anchor_id, record_hash, transaction_hash, block_number, timestamp
- F5.5: Event replay: replay historical events from a specific block number for reindexing
- F5.6: Event filtering: filter events by type, party, time range, contract address
- F5.7: Webhook notifications: notify external systems when specific on-chain events occur
- F5.8: Event persistence: store indexed events with full blockchain context
- F5.9: Chain reorganization handling: detect and handle chain reorgs by re-processing affected events
- F5.10: Multi-chain event aggregation: aggregate events from multiple chains into unified index

### 6.6 Feature 6: Merkle Proof Generator (P0)

**Requirements**:
- F6.1: Construct Merkle trees from batches of record hashes using SHA-256
- F6.2: Generate compact inclusion proofs for individual records within a batch
- F6.3: Proof serialization: JSON and binary formats for efficient transmission
- F6.4: Proof verification: standalone verification using only proof, record hash, and Merkle root
- F6.5: Batch sizes: support trees with 1 to 10,000 leaves
- F6.6: Sorted tree construction: deterministic ordering of leaves for reproducible roots
- F6.7: Proof size optimization: O(log n) proof path length
- F6.8: Multi-proof generation: generate proofs for multiple records in a single batch efficiently
- F6.9: Incremental tree updates: support appending new records to existing trees
- F6.10: Tree serialization: export/import complete Merkle trees for archival

### 6.7 Feature 7: Cross-Party Data Sharing (P0)

**Requirements**:
- F7.1: On-chain access control: grant/revoke read access to specific records for specific parties
- F7.2: Permissioned sharing: operator controls which data is shared with which parties
- F7.3: Competent authority access: special access level for regulatory authorities
- F7.4: Auditor access: time-limited read access for third-party auditors
- F7.5: Shared record types: DDS summaries, custody transfers, certification references, verification results
- F7.6: Privacy preservation: only hashes are on-chain; actual data shared through secure off-chain channels
- F7.7: Access audit trail: on-chain record of all access grants and revocations
- F7.8: Multi-party confirmation: require multiple parties to confirm a custody transfer before anchoring
- F7.9: Dispute resolution: on-chain dispute filing when parties disagree on record content
- F7.10: Data request workflow: parties can request access to specific records through on-chain requests

### 6.8 Feature 8: Compliance Evidence Packager (P0)

**Requirements**:
- F8.1: Generate blockchain-backed evidence packages for regulatory submissions
- F8.2: Package contents: record data, SHA-256 hash, blockchain anchor receipt, Merkle proof, verification status
- F8.3: Evidence chain: link evidence packages across the full supply chain (plot -> product)
- F8.4: Package formats: JSON, PDF (with embedded verification data), EUDR XML
- F8.5: Package verification: self-verifying packages that contain all data needed for independent verification
- F8.6: Regulatory report: summary of all anchored records for a specific DDS or time period
- F8.7: Compliance timeline: chronological view of all anchored events for a supply chain
- F8.8: Evidence completeness check: verify that all required records for a DDS have been anchored
- F8.9: Package signing: sign evidence packages with operator's private key for non-repudiation
- F8.10: 5-year package retention per EUDR Article 14

---

## 7. Technical Requirements

### 7.1 Architecture

```
greenlang/agents/eudr/blockchain_integration/
    __init__.py                          # Package exports (80+ symbols)
    config.py                            # BlockchainIntegrationConfig singleton
    models.py                            # Pydantic v2 models, enums
    provenance.py                        # SHA-256 chain hashing
    metrics.py                           # Prometheus metrics (gl_eudr_bci_ prefix)
    transaction_anchor.py                # Engine 1: Transaction anchoring
    smart_contract_manager.py            # Engine 2: Smart contract deployment/interaction
    multi_chain_connector.py             # Engine 3: Multi-chain abstraction
    verification_engine.py               # Engine 4: On-chain verification
    event_listener.py                    # Engine 5: Event listening and indexing
    merkle_proof_generator.py            # Engine 6: Merkle tree and proof generation
    cross_party_sharing.py               # Engine 7: Cross-party data sharing
    compliance_evidence_packager.py      # Engine 8: Evidence package generation
    setup.py                             # BlockchainIntegrationService facade
    reference_data/
        __init__.py
        chain_configs.py                 # Network configurations (Ethereum, Polygon, Fabric, Besu)
        contract_abis.py                 # Smart contract ABI definitions
        anchor_rules.py                  # Anchoring rules per event type
    api/
        __init__.py
        router.py
        schemas.py
        dependencies.py
        anchor_routes.py                 # Transaction anchoring endpoints
        contract_routes.py               # Smart contract management endpoints
        chain_routes.py                  # Multi-chain connection endpoints
        verification_routes.py           # On-chain verification endpoints
        event_routes.py                  # Event listening endpoints
        merkle_routes.py                 # Merkle proof endpoints
        sharing_routes.py                # Cross-party sharing endpoints
        evidence_routes.py               # Compliance evidence endpoints
```

### 7.2 Database Schema (V101)

| Table | Type | Description |
|-------|------|-------------|
| `gl_eudr_bci_anchors` | hypertable (monthly) | Transaction anchor records with blockchain references |
| `gl_eudr_bci_merkle_trees` | regular | Merkle tree definitions with root hashes |
| `gl_eudr_bci_merkle_leaves` | regular | Individual leaf records within Merkle trees |
| `gl_eudr_bci_smart_contracts` | regular | Deployed smart contract registry |
| `gl_eudr_bci_contract_events` | hypertable (monthly) | Indexed on-chain events |
| `gl_eudr_bci_chain_connections` | regular | Blockchain network connection configurations |
| `gl_eudr_bci_verification_results` | hypertable (monthly) | On-chain verification results |
| `gl_eudr_bci_access_grants` | regular | Cross-party access control records |
| `gl_eudr_bci_evidence_packages` | regular | Generated compliance evidence packages |
| `gl_eudr_bci_gas_costs` | regular | Gas cost tracking per transaction |
| `gl_eudr_bci_batch_jobs` | regular | Batch anchoring jobs |
| `gl_eudr_bci_audit_log` | regular | Immutable audit trail |

### 7.3 Prometheus Metrics (18 metrics, `gl_eudr_bci_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_bci_anchors_total` | Counter | Total anchors created |
| `gl_eudr_bci_anchors_confirmed_total` | Counter | Anchors confirmed on-chain |
| `gl_eudr_bci_anchors_failed_total` | Counter | Failed anchor attempts |
| `gl_eudr_bci_verifications_total` | Counter | On-chain verifications performed |
| `gl_eudr_bci_verifications_tampered_total` | Counter | Tampered records detected |
| `gl_eudr_bci_merkle_trees_total` | Counter | Merkle trees constructed |
| `gl_eudr_bci_merkle_proofs_total` | Counter | Merkle proofs generated |
| `gl_eudr_bci_events_indexed_total` | Counter | On-chain events indexed |
| `gl_eudr_bci_contracts_deployed_total` | Counter | Smart contracts deployed |
| `gl_eudr_bci_access_grants_total` | Counter | Access grants issued |
| `gl_eudr_bci_evidence_packages_total` | Counter | Evidence packages generated |
| `gl_eudr_bci_gas_spent_total` | Counter | Total gas spent (wei) |
| `gl_eudr_bci_api_errors_total` | Counter | API errors |
| `gl_eudr_bci_anchor_duration_seconds` | Histogram | Anchor confirmation latency |
| `gl_eudr_bci_verification_duration_seconds` | Histogram | Verification processing latency |
| `gl_eudr_bci_merkle_build_duration_seconds` | Histogram | Merkle tree construction latency |
| `gl_eudr_bci_active_listeners` | Gauge | Active event listeners |
| `gl_eudr_bci_pending_anchors` | Gauge | Anchors pending confirmation |

### 7.4 API Endpoints (~37 endpoints)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| Anchor | POST | `/api/v1/eudr-bci/anchors` | Create anchor (single record) |
| | POST | `/api/v1/eudr-bci/anchors/batch` | Batch anchor (multiple records) |
| | GET | `/api/v1/eudr-bci/anchors/{anchor_id}` | Get anchor details |
| | GET | `/api/v1/eudr-bci/anchors/status/{tx_hash}` | Get anchor status by tx hash |
| | GET | `/api/v1/eudr-bci/anchors/history/{record_id}` | Get anchor history for record |
| Contract | POST | `/api/v1/eudr-bci/contracts/deploy` | Deploy smart contract |
| | GET | `/api/v1/eudr-bci/contracts/{contract_id}` | Get contract details |
| | POST | `/api/v1/eudr-bci/contracts/{contract_id}/call` | Call contract method |
| | GET | `/api/v1/eudr-bci/contracts/{contract_id}/state` | Get contract state |
| | GET | `/api/v1/eudr-bci/contracts` | List deployed contracts |
| Chain | POST | `/api/v1/eudr-bci/chains/connect` | Connect to blockchain network |
| | GET | `/api/v1/eudr-bci/chains/{chain_id}/status` | Get chain connection status |
| | GET | `/api/v1/eudr-bci/chains` | List connected chains |
| | POST | `/api/v1/eudr-bci/chains/{chain_id}/estimate-gas` | Estimate gas for operation |
| Verify | POST | `/api/v1/eudr-bci/verify` | Verify record against on-chain anchor |
| | POST | `/api/v1/eudr-bci/verify/batch` | Batch verify records |
| | POST | `/api/v1/eudr-bci/verify/merkle-proof` | Verify Merkle inclusion proof |
| | GET | `/api/v1/eudr-bci/verify/{verification_id}` | Get verification result |
| Event | POST | `/api/v1/eudr-bci/events/subscribe` | Subscribe to on-chain events |
| | DELETE | `/api/v1/eudr-bci/events/subscribe/{subscription_id}` | Unsubscribe |
| | GET | `/api/v1/eudr-bci/events` | Query indexed events |
| | GET | `/api/v1/eudr-bci/events/{event_id}` | Get event details |
| | POST | `/api/v1/eudr-bci/events/replay` | Replay events from block |
| Merkle | POST | `/api/v1/eudr-bci/merkle/build` | Build Merkle tree |
| | GET | `/api/v1/eudr-bci/merkle/{tree_id}` | Get Merkle tree |
| | POST | `/api/v1/eudr-bci/merkle/{tree_id}/proof` | Generate inclusion proof |
| | POST | `/api/v1/eudr-bci/merkle/verify` | Verify Merkle proof |
| Sharing | POST | `/api/v1/eudr-bci/sharing/grant` | Grant access to party |
| | DELETE | `/api/v1/eudr-bci/sharing/revoke/{grant_id}` | Revoke access |
| | GET | `/api/v1/eudr-bci/sharing/grants/{record_id}` | List access grants |
| | POST | `/api/v1/eudr-bci/sharing/request` | Request access to record |
| | POST | `/api/v1/eudr-bci/sharing/confirm` | Multi-party confirmation |
| Evidence | POST | `/api/v1/eudr-bci/evidence/package` | Generate evidence package |
| | GET | `/api/v1/eudr-bci/evidence/{package_id}` | Get evidence package |
| | GET | `/api/v1/eudr-bci/evidence/{package_id}/download` | Download evidence package |
| | POST | `/api/v1/eudr-bci/evidence/verify` | Verify evidence package |
| Batch | POST | `/api/v1/eudr-bci/batch` | Submit batch job |
| | DELETE | `/api/v1/eudr-bci/batch/{job_id}` | Cancel batch job |
| Health | GET | `/api/v1/eudr-bci/health` | Health check |

---

## 8. Test Strategy

### 8.1 Unit Tests (500+)

- Transaction anchoring for all 8 event types with hash computation
- Merkle tree construction with 1, 2, 100, 1000, 10000 leaves
- Merkle proof generation and verification (valid/invalid/tampered)
- Smart contract ABI encoding/decoding for all contract types
- Multi-chain connector for Ethereum, Polygon, Fabric, Besu protocols
- On-chain verification with matching/mismatching hashes
- Event listener subscription, filtering, and indexing
- Cross-party access control: grant, revoke, query, multi-party confirmation
- Compliance evidence package generation in all formats
- Gas cost estimation and tracking
- Edge cases: empty batches, maximum tree depth, network timeouts, nonce conflicts
- Deterministic Merkle root reproducibility tests

### 8.2 Performance Tests

- Batch anchoring of 10,000 records into single Merkle tree
- Concurrent verification of 1,000 records
- Event replay of 100,000 historical events
- Merkle proof generation for 10,000-leaf tree

---

## Appendices

### A. Smart Contract Specifications

| Contract | Purpose | Key Functions |
|----------|---------|---------------|
| AnchorRegistry | Store Merkle roots with metadata | `anchor(bytes32 root, uint256 count)`, `verify(bytes32 root)`, `getAnchor(uint256 id)` |
| CustodyTransfer | Record custody transfers | `recordTransfer(bytes32 hash, address from, address to)`, `confirmTransfer(uint256 id)` |
| ComplianceCheck | Enforce compliance rules | `checkCompliance(bytes32 ddsHash)`, `registerParty(address party)` |

### B. Anchor Event Type Configuration

| Event Type | Priority | Batch Eligible | Max Batch Wait | Immediate Anchor |
|------------|----------|----------------|----------------|-----------------|
| DDS Submission | P0 | Yes | 5 min | Optional |
| Custody Transfer | P0 | Yes | 5 min | Optional |
| Batch Event | P0 | Yes | 10 min | No |
| Certificate Reference | P1 | Yes | 15 min | No |
| Reconciliation Result | P1 | Yes | 30 min | No |
| Mass Balance Entry | P2 | Yes | 60 min | No |
| Document Authentication | P2 | Yes | 60 min | No |
| Geolocation Verification | P2 | Yes | 60 min | No |

### C. Gas Cost Estimates (Polygon)

| Operation | Estimated Gas | Estimated Cost (MATIC) |
|-----------|--------------|----------------------|
| Single anchor (AnchorRegistry) | ~65,000 | ~0.001 |
| Batch anchor (100 records, 1 Merkle root) | ~85,000 | ~0.0013 |
| Custody transfer recording | ~120,000 | ~0.0018 |
| Party registration | ~45,000 | ~0.0007 |
| Access grant | ~55,000 | ~0.0008 |

### D. Chain Configuration Reference

| Parameter | Ethereum | Polygon | Fabric | Besu |
|-----------|----------|---------|--------|------|
| Consensus | PoS | PoS | PBFT/Raft | IBFT2/QBFT |
| Block Time | ~12s | ~2s | Configurable | Configurable |
| Confirmation Depth | 12 blocks | 32 blocks | 1 block | 1 block |
| Transaction Format | EIP-1559 | EIP-1559 | Protobuf | EIP-1559 |
| Smart Contract Language | Solidity | Solidity | Go/Node.js | Solidity |
| Event System | Logs/Topics | Logs/Topics | ChaincodeEvents | Logs/Topics |
