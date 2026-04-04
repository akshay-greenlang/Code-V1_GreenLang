# AGENT-EUDR-013: Blockchain Integration API

**Agent ID:** `GL-EUDR-BCI-013`
**Prefix:** `/v1/eudr-bci`
**Version:** 1.0.0
**PRD:** AGENT-EUDR-013
**Regulation:** EU 2023/1115 (EUDR) -- Immutable audit trail per Article 31

## Purpose

The Blockchain Integration agent provides immutable, tamper-proof anchoring of
EUDR compliance evidence onto distributed ledger networks. It anchors
transaction hashes, manages smart contracts, connects to multiple blockchain
networks (Ethereum, Polygon, Hyperledger), verifies on-chain evidence,
listens for blockchain events, generates Merkle proofs, facilitates
cross-party data sharing, and packages on-chain evidence for compliance.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/anchors` | Anchor hash to blockchain | JWT |
| GET | `/anchors/{anchor_id}` | Get anchor details | JWT |
| POST | `/anchors/batch` | Batch anchor hashes | JWT |
| GET | `/anchors/verify/{anchor_id}` | Verify anchor on-chain | JWT |
| GET | `/anchors/history` | List anchor history | JWT |
| POST | `/contracts/deploy` | Deploy smart contract | JWT |
| GET | `/contracts` | List contracts | JWT |
| GET | `/contracts/{contract_id}` | Get contract details | JWT |
| POST | `/contracts/{contract_id}/execute` | Execute contract method | JWT |
| GET | `/contracts/{contract_id}/events` | Get contract events | JWT |
| POST | `/chains/connect` | Connect to blockchain | JWT |
| GET | `/chains` | List connected chains | JWT |
| GET | `/chains/{chain_id}` | Get chain details | JWT |
| GET | `/chains/{chain_id}/status` | Get chain health | JWT |
| POST | `/verify` | Verify on-chain hash | JWT |
| GET | `/verify/{verification_id}` | Get verification result | JWT |
| POST | `/verify/batch` | Batch on-chain verify | JWT |
| POST | `/verify/certificate` | Verify certificate on-chain | JWT |
| POST | `/events/subscribe` | Subscribe to events | JWT |
| GET | `/events/subscriptions` | List subscriptions | JWT |
| DELETE | `/events/subscriptions/{sub_id}` | Remove subscription | JWT |
| GET | `/events/history` | Get event history | JWT |
| POST | `/events/replay` | Replay events | JWT |
| POST | `/merkle/build` | Build Merkle tree | JWT |
| POST | `/merkle/proof` | Generate Merkle proof | JWT |
| POST | `/merkle/verify` | Verify Merkle proof | JWT |
| GET | `/merkle/{tree_id}` | Get Merkle tree | JWT |
| POST | `/sharing/share` | Share data cross-party | JWT |
| GET | `/sharing/shared` | List shared items | JWT |
| POST | `/sharing/accept` | Accept shared data | JWT |
| POST | `/sharing/revoke` | Revoke sharing | JWT |
| GET | `/sharing/{share_id}` | Get share details | JWT |
| POST | `/evidence/package` | Create evidence package | JWT |
| GET | `/evidence/{package_id}` | Get evidence package | JWT |
| POST | `/evidence/verify` | Verify evidence package | JWT |
| GET | `/evidence/history` | Get evidence history | JWT |
| POST | `/batch` | Submit batch job | JWT |
| DELETE | `/batch/{job_id}` | Cancel batch job | JWT |
| GET | `/health` | Health check | None |

**Total: 39 endpoints**

---

## Endpoints

### POST /v1/eudr-bci/anchors

Anchor a SHA-256 hash onto a configured blockchain network, creating an
immutable timestamp proof for EUDR Article 31 audit trail compliance.

**Request:**

```json
{
  "hash": "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  "entity_type": "due_diligence_statement",
  "entity_id": "DDS-2026-001",
  "chain": "polygon",
  "metadata": {
    "operator_id": "OP-2024-001",
    "commodity": "cocoa"
  }
}
```

**Response (201 Created):**

```json
{
  "anchor_id": "anc_001",
  "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
  "chain": "polygon",
  "block_number": 45678901,
  "block_timestamp": "2026-04-04T10:00:15Z",
  "confirmation_count": 12,
  "status": "confirmed",
  "cost_gwei": 21000,
  "anchored_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-bci/merkle/proof

Generate a Merkle proof for a specific leaf in a previously built Merkle tree,
proving that a document hash is included in a set of hashes without revealing
the other hashes.

**Request:**

```json
{
  "tree_id": "mktree_001",
  "leaf_hash": "sha256:a1b2c3d4..."
}
```

**Response (200 OK):**

```json
{
  "proof_id": "mkprf_001",
  "tree_id": "mktree_001",
  "leaf_hash": "sha256:a1b2c3d4...",
  "root_hash": "sha256:r1s2t3u4...",
  "proof_path": [
    {"hash": "sha256:x1y2z3...", "position": "right"},
    {"hash": "sha256:p1q2r3...", "position": "left"}
  ],
  "leaf_index": 3,
  "tree_size": 8,
  "verified": true
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_hash` | Hash format is invalid |
| 404 | `anchor_not_found` | Anchor ID not found on-chain |
| 409 | `job_not_cancellable` | Batch job already completed |
| 503 | `chain_unavailable` | Blockchain network is unreachable |
