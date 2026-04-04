# Duplicate Detection API Reference

**Agent:** AGENT-DATA-011 (GL-DATA-X-014)
**Prefix:** `/api/v1/dedup`
**Source:** `greenlang/agents/data/duplicate_detector/api/router.py`
**Status:** Production Ready

## Overview

The Duplicate Detection agent provides a full record deduplication pipeline: fingerprinting, blocking, pairwise comparison, match classification, cluster formation, and golden record merging. Supports multiple algorithms (SHA-256, SimHash, MinHash) and merge strategies.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/jobs` | Create dedup job | Yes |
| 2 | GET | `/jobs` | List jobs | Yes |
| 3 | GET | `/jobs/{job_id}` | Get job details | Yes |
| 4 | DELETE | `/jobs/{job_id}` | Cancel job | Yes |
| 5 | POST | `/fingerprint` | Fingerprint records | Yes |
| 6 | POST | `/block` | Create blocks | Yes |
| 7 | POST | `/compare` | Compare pairs | Yes |
| 8 | POST | `/classify` | Classify matches | Yes |
| 9 | GET | `/matches` | List matches | Yes |
| 10 | GET | `/matches/{match_id}` | Get match details | Yes |
| 11 | POST | `/clusters` | Form clusters | Yes |
| 12 | GET | `/clusters` | List clusters | Yes |
| 13 | GET | `/clusters/{cluster_id}` | Get cluster details | Yes |
| 14 | POST | `/merge` | Execute merge | Yes |
| 15 | GET | `/merge/{merge_id}` | Get merge result | Yes |
| 16 | POST | `/pipeline` | Run full pipeline | Yes |
| 17 | POST | `/rules` | Create dedup rule | Yes |
| 18 | GET | `/rules` | List rules | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Statistics | Yes |

---

## Key Endpoints

### 16. Run Full Pipeline

Runs the entire deduplication pipeline end-to-end (fingerprint, block, compare, classify, cluster, merge) in a single call.

```http
POST /api/v1/dedup/pipeline
```

**Request Body:**

```json
{
  "records": [
    {"id": "1", "name": "Acme Corp", "address": "123 Main St", "city": "Springfield"},
    {"id": "2", "name": "ACME Corporation", "address": "123 Main Street", "city": "Springfield"},
    {"id": "3", "name": "Widget Inc", "address": "456 Oak Ave", "city": "Portland"}
  ],
  "rule": {
    "blocking_strategy": "sorted_neighborhood",
    "blocking_key_fields": ["city"],
    "match_threshold": 0.85,
    "possible_threshold": 0.65,
    "merge_strategy": "golden_record"
  },
  "options": {
    "fingerprint_algorithm": "simhash",
    "clustering_algorithm": "union_find"
  }
}
```

**Response:**

```json
{
  "pipeline_id": "pipe_abc123",
  "status": "completed",
  "input_count": 3,
  "duplicate_clusters": 1,
  "unique_records": 2,
  "merged_records": [
    {
      "golden_record_id": "gr_001",
      "name": "Acme Corp",
      "address": "123 Main St",
      "city": "Springfield",
      "source_ids": ["1", "2"],
      "confidence": 0.92
    },
    {
      "golden_record_id": "gr_002",
      "name": "Widget Inc",
      "address": "456 Oak Ave",
      "city": "Portland",
      "source_ids": ["3"],
      "confidence": 1.0
    }
  ],
  "provenance_hash": "sha256:..."
}
```

### 5. Fingerprint Records

```http
POST /api/v1/dedup/fingerprint
```

**Request Body:**

```json
{
  "records": [{"name": "Acme Corp", "city": "Springfield"}],
  "field_set": ["name", "city"],
  "algorithm": "sha256"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `algorithm` | string | `sha256`, `simhash`, `minhash` |
| `field_set` | string[] | Fields to include in fingerprint (all if omitted) |

### 14. Execute Merge

```http
POST /api/v1/dedup/merge
```

| Field | Type | Description |
|-------|------|-------------|
| `strategy` | string | `keep_first`, `keep_latest`, `keep_most_complete`, `merge_fields`, `golden_record` |
