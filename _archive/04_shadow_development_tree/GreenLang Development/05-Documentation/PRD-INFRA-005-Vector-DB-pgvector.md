# PRD-INFRA-005: Vector Database Infrastructure with pgvector

**Document Version:** 1.0.0
**Created:** 2026-02-03
**Last Updated:** 2026-02-03
**Author:** GreenLang Platform Engineering
**Status:** Approved for Development
**Priority:** P1 - Critical Infrastructure

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals and Objectives](#3-goals-and-objectives)
4. [Technical Architecture](#4-technical-architecture)
5. [Component Specifications](#5-component-specifications)
6. [Database Schema Design](#6-database-schema-design)
7. [Embedding Pipeline](#7-embedding-pipeline)
8. [Query Optimization](#8-query-optimization)
9. [Scaling Strategy](#9-scaling-strategy)
10. [Security Requirements](#10-security-requirements)
11. [Monitoring and Observability](#11-monitoring-and-observability)
12. [Backup and Recovery](#12-backup-and-recovery)
13. [Migration Strategy](#13-migration-strategy)
14. [Performance Requirements](#14-performance-requirements)
15. [Cost Analysis](#15-cost-analysis)
16. [Implementation Roadmap](#16-implementation-roadmap)
17. [Success Criteria](#17-success-criteria)
18. [Appendices](#18-appendices)

---

## 1. Executive Summary

### 1.1 Overview

This PRD defines the infrastructure requirements for deploying pgvector extension on PostgreSQL/Aurora to provide native vector embedding storage and similarity search capabilities for the GreenLang platform. This infrastructure enables AI-powered semantic search, document retrieval, and embedding-based analytics across all GreenLang applications.

### 1.2 Business Value

| Benefit | Impact |
|---------|--------|
| **Unified Database** | Single PostgreSQL instance for relational + vector data |
| **Cost Reduction** | 40-60% lower than dedicated vector databases |
| **Simplified Operations** | Leverage existing PostgreSQL expertise |
| **ACID Compliance** | Full transactional support for embeddings |
| **Regulatory Alignment** | EU data residency with existing RDS infrastructure |

### 1.3 Scope

**In Scope:**
- pgvector extension deployment on Aurora PostgreSQL
- Vector embedding storage schemas
- HNSW and IVFFlat index configurations
- Embedding generation pipeline integration
- Similarity search query optimization
- Monitoring dashboards and alerts
- Backup and recovery procedures
- Migration from Weaviate (optional hybrid)

**Out of Scope:**
- Weaviate replacement (hybrid approach supported)
- Custom embedding model training
- Real-time streaming embeddings
- Multi-region vector replication (Phase 2)

### 1.4 Key Stakeholders

| Role | Responsibility |
|------|----------------|
| Platform Engineering | Infrastructure deployment and operations |
| ML Engineering | Embedding pipeline and model integration |
| Application Teams | Vector search integration |
| Security Team | Data protection and compliance |
| SRE Team | Monitoring and incident response |

---

## 2. Problem Statement

### 2.1 Current State

The GreenLang platform currently uses:
- **Weaviate** (K8s StatefulSet) for GL-VCCI-APP vector storage
- **FAISS** (in-memory) for deterministic search operations
- **No pgvector** configured in PostgreSQL despite planned usage

### 2.2 Challenges

1. **Operational Complexity**: Managing separate vector database (Weaviate) increases operational burden
2. **Data Consistency**: No transactional guarantees between relational and vector data
3. **Cost Overhead**: Dedicated Weaviate cluster adds infrastructure costs
4. **Limited Integration**: FAISS in-memory storage doesn't persist across restarts
5. **Scaling Issues**: Weaviate requires separate scaling considerations

### 2.3 Requirements Gap

| Requirement | Current State | Target State |
|-------------|---------------|--------------|
| Vector Storage | Weaviate (separate) | pgvector (integrated) |
| ACID Transactions | Not supported | Full support |
| Backup Integration | Separate process | Unified with RDS |
| Query Optimization | Limited | PostgreSQL optimizer |
| Cost per Million Vectors | $150/month | $50/month |

---

## 3. Goals and Objectives

### 3.1 Primary Goals

1. **Deploy pgvector extension** on Aurora PostgreSQL cluster
2. **Enable vector similarity search** with sub-100ms latency
3. **Support 100M+ vector embeddings** with efficient indexing
4. **Integrate with existing RAG system** seamlessly
5. **Maintain 99.9% availability** for vector operations

### 3.2 Key Results

| Objective | Key Result | Target |
|-----------|------------|--------|
| Performance | P99 similarity search latency | < 100ms |
| Scale | Supported vector count | 100M embeddings |
| Availability | Vector search uptime | 99.9% |
| Cost | Monthly infrastructure cost | < $500/env |
| Integration | RAG system migration | 100% compatible |

### 3.3 Non-Goals

- Replace Weaviate entirely (hybrid approach supported)
- Build custom vector database
- Support non-PostgreSQL databases
- Real-time embedding streaming (batch supported)

---

## 4. Technical Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GreenLang Vector Infrastructure                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   Application   │    │   Embedding     │    │   Vector        │          │
│  │   Services      │    │   Service       │    │   Search API    │          │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘          │
│           │                      │                      │                    │
│           ▼                      ▼                      ▼                    │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Vector Operations Layer                        │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │       │
│  │  │  Embed     │  │  Index     │  │  Search    │  │  Batch     │ │       │
│  │  │  Manager   │  │  Manager   │  │  Engine    │  │  Processor │ │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                     Aurora PostgreSQL + pgvector                  │       │
│  │  ┌─────────────────────────────────────────────────────────────┐ │       │
│  │  │                    Primary Instance                          │ │       │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │ │       │
│  │  │  │  pgvector   │  │  HNSW       │  │  IVFFlat    │         │ │       │
│  │  │  │  Extension  │  │  Indexes    │  │  Indexes    │         │ │       │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘         │ │       │
│  │  └─────────────────────────────────────────────────────────────┘ │       │
│  │                              │                                    │       │
│  │                              ▼                                    │       │
│  │  ┌─────────────────────────────────────────────────────────────┐ │       │
│  │  │                    Read Replicas (2x)                        │ │       │
│  │  │         (Vector search queries distributed here)             │ │       │
│  │  └─────────────────────────────────────────────────────────────┘ │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AWS Infrastructure                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                         VPC (10.0.0.0/16)                         │       │
│  │  ┌────────────────────────────────────────────────────────────┐  │       │
│  │  │                    Private Subnets                          │  │       │
│  │  │                                                              │  │       │
│  │  │  ┌─────────────────────────────────────────────────────┐   │  │       │
│  │  │  │              Aurora PostgreSQL Cluster               │   │  │       │
│  │  │  │                                                       │   │  │       │
│  │  │  │  ┌───────────────┐  ┌───────────────┐               │   │  │       │
│  │  │  │  │   Primary     │  │   Reader 1    │               │   │  │       │
│  │  │  │  │   (Writer)    │  │   (us-east-1b)│               │   │  │       │
│  │  │  │  │  us-east-1a   │  │   pgvector    │               │   │  │       │
│  │  │  │  │  pgvector 0.7 │  │   read-only   │               │   │  │       │
│  │  │  │  │  r6g.2xlarge  │  │   r6g.xlarge  │               │   │  │       │
│  │  │  │  └───────────────┘  └───────────────┘               │   │  │       │
│  │  │  │                                                       │   │  │       │
│  │  │  │                     ┌───────────────┐               │   │  │       │
│  │  │  │                     │   Reader 2    │               │   │  │       │
│  │  │  │                     │   (us-east-1c)│               │   │  │       │
│  │  │  │                     │   pgvector    │               │   │  │       │
│  │  │  │                     │   read-only   │               │   │  │       │
│  │  │  │                     │   r6g.xlarge  │               │   │  │       │
│  │  │  │                     └───────────────┘               │   │  │       │
│  │  │  └─────────────────────────────────────────────────────┘   │  │       │
│  │  │                                                              │  │       │
│  │  │  ┌─────────────────────────────────────────────────────┐   │  │       │
│  │  │  │                   EKS Cluster                        │   │  │       │
│  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │  │       │
│  │  │  │  │  Embedding  │  │  Vector     │  │  RAG        │ │   │  │       │
│  │  │  │  │  Service    │  │  Search     │  │  Engine     │ │   │  │       │
│  │  │  │  │  (3 pods)   │  │  API (3)    │  │  (3 pods)   │ │   │  │       │
│  │  │  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │  │       │
│  │  │  └─────────────────────────────────────────────────────┘   │  │       │
│  │  └────────────────────────────────────────────────────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Vector Data Flow                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  INGESTION FLOW:                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐      │
│  │  Source  │───▶│  Chunker │───▶│  Embed   │───▶│  pgvector        │      │
│  │  Document│    │  Service │    │  Model   │    │  INSERT + INDEX  │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────────┘      │
│       │                                                    │                 │
│       │              Batch: 1000 chunks/batch              │                 │
│       │              Latency: < 5s per batch               │                 │
│       └────────────────────────────────────────────────────┘                 │
│                                                                               │
│  QUERY FLOW:                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐    ┌──────────┐      │
│  │  Query   │───▶│  Embed   │───▶│  pgvector        │───▶│  Results │      │
│  │  Text    │    │  Query   │    │  Similarity      │    │  Ranked  │      │
│  └──────────┘    └──────────┘    │  Search (HNSW)   │    └──────────┘      │
│       │                          └──────────────────┘           │            │
│       │              Target: < 50ms E2E                         │            │
│       └─────────────────────────────────────────────────────────┘            │
│                                                                               │
│  HYBRID SEARCH FLOW:                                                         │
│  ┌──────────┐    ┌──────────────────────────────────┐    ┌──────────┐      │
│  │  Query   │───▶│  Parallel Execution              │───▶│  RRF     │      │
│  │  Text    │    │  ┌────────────┐  ┌────────────┐ │    │  Fusion  │      │
│  └──────────┘    │  │  Vector    │  │  Full-Text │ │    └──────────┘      │
│                  │  │  Search    │  │  Search    │ │           │            │
│                  │  └────────────┘  └────────────┘ │           │            │
│                  └──────────────────────────────────┘           │            │
│                        Target: < 100ms E2E                      │            │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Specifications

### 5.1 pgvector Extension

| Specification | Value |
|---------------|-------|
| **Extension Version** | 0.7.0+ |
| **PostgreSQL Version** | 15.x (Aurora) |
| **Vector Dimensions** | 384 (MiniLM), 768 (MPNet), 1536 (OpenAI) |
| **Distance Functions** | cosine, L2 (euclidean), inner product |
| **Index Types** | HNSW (primary), IVFFlat (fallback) |

### 5.2 Index Configurations

#### 5.2.1 HNSW Index (Primary)

```sql
-- HNSW Index for semantic search
-- Optimized for recall and latency
CREATE INDEX CONCURRENTLY idx_embeddings_hnsw
ON vector_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,                    -- Connections per layer (default: 16)
    ef_construction = 200,     -- Build-time search width (higher = better recall)
    ef_search = 100            -- Query-time search width (configurable per query)
);
```

**HNSW Tuning Parameters:**

| Parameter | Dev | Staging | Production | Description |
|-----------|-----|---------|------------|-------------|
| `m` | 8 | 16 | 24 | Max connections per node |
| `ef_construction` | 100 | 200 | 400 | Build-time beam width |
| `ef_search` | 40 | 100 | 200 | Query-time beam width |

#### 5.2.2 IVFFlat Index (Bulk Operations)

```sql
-- IVFFlat Index for bulk similarity operations
-- Faster to build, good for large datasets
CREATE INDEX CONCURRENTLY idx_embeddings_ivfflat
ON vector_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);  -- sqrt(n) where n = expected row count
```

### 5.3 Embedding Models

| Model | Dimensions | Use Case | Latency |
|-------|------------|----------|---------|
| `all-MiniLM-L6-v2` | 384 | Default semantic search | 5ms |
| `all-mpnet-base-v2` | 768 | High-quality retrieval | 15ms |
| `text-embedding-3-small` | 1536 | OpenAI compatibility | 50ms |

### 5.4 Hardware Specifications

#### 5.4.1 Aurora Instance Sizing

| Environment | Instance Type | vCPU | Memory | Storage | IOPS |
|-------------|---------------|------|--------|---------|------|
| Development | db.r6g.large | 2 | 16 GB | 100 GB | 3000 |
| Staging | db.r6g.xlarge | 4 | 32 GB | 500 GB | 6000 |
| Production | db.r6g.2xlarge | 8 | 64 GB | 2 TB | 12000 |

#### 5.4.2 Memory Requirements

```
Vector Memory Formula:
Memory = (vectors × dimensions × 4 bytes) + (index overhead × 1.5)

Example for 100M 384-dim vectors:
- Raw vectors: 100M × 384 × 4 = 153.6 GB
- HNSW overhead: ~1.5x = 230 GB total
- Recommended instance: db.r6g.4xlarge (128 GB) + Aurora storage
```

---

## 6. Database Schema Design

### 6.1 Core Tables

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Vector embeddings table
CREATE TABLE vector_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source reference
    source_type VARCHAR(50) NOT NULL,  -- 'document', 'regulation', 'report', 'policy'
    source_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,

    -- Content
    content_hash VARCHAR(64) NOT NULL,  -- SHA-256 of content
    content_preview TEXT,               -- First 500 chars

    -- Vector embedding
    embedding vector(384) NOT NULL,     -- MiniLM default
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'all-MiniLM-L6-v2',

    -- Metadata
    metadata JSONB NOT NULL DEFAULT '{}',
    namespace VARCHAR(100) NOT NULL DEFAULT 'default',

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_source_chunk UNIQUE (source_type, source_id, chunk_index, embedding_model)
);

-- Partitioning by namespace for multi-tenant isolation
CREATE TABLE vector_embeddings_partitioned (
    LIKE vector_embeddings INCLUDING ALL
) PARTITION BY LIST (namespace);

-- Create partitions per application
CREATE TABLE vector_embeddings_csrd PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('csrd');
CREATE TABLE vector_embeddings_cbam PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('cbam');
CREATE TABLE vector_embeddings_eudr PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('eudr');
CREATE TABLE vector_embeddings_vcci PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('vcci');
CREATE TABLE vector_embeddings_default PARTITION OF vector_embeddings_partitioned
    DEFAULT;
```

### 6.2 Supporting Tables

```sql
-- Embedding collections (logical groupings)
CREATE TABLE embedding_collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL UNIQUE,
    description TEXT,
    namespace VARCHAR(100) NOT NULL DEFAULT 'default',
    embedding_model VARCHAR(100) NOT NULL,
    dimensions INTEGER NOT NULL,
    distance_metric VARCHAR(20) NOT NULL DEFAULT 'cosine',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Embedding generation jobs
CREATE TABLE embedding_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID REFERENCES embedding_collections(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    source_type VARCHAR(50) NOT NULL,
    source_count INTEGER NOT NULL DEFAULT 0,
    processed_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Search analytics
CREATE TABLE vector_search_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_embedding vector(384),
    query_text TEXT,
    namespace VARCHAR(100),
    top_k INTEGER,
    threshold FLOAT,
    result_count INTEGER,
    latency_ms INTEGER,
    user_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 6.3 Indexes

```sql
-- HNSW index for similarity search (cosine distance)
CREATE INDEX CONCURRENTLY idx_embeddings_hnsw_cosine
ON vector_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Filtered search indexes
CREATE INDEX CONCURRENTLY idx_embeddings_namespace
ON vector_embeddings (namespace);

CREATE INDEX CONCURRENTLY idx_embeddings_source
ON vector_embeddings (source_type, source_id);

CREATE INDEX CONCURRENTLY idx_embeddings_created
ON vector_embeddings (created_at DESC);

-- Metadata GIN index for JSON filtering
CREATE INDEX CONCURRENTLY idx_embeddings_metadata
ON vector_embeddings USING GIN (metadata jsonb_path_ops);

-- Partial indexes per namespace for faster filtered queries
CREATE INDEX CONCURRENTLY idx_embeddings_csrd_hnsw
ON vector_embeddings
USING hnsw (embedding vector_cosine_ops)
WHERE namespace = 'csrd';

CREATE INDEX CONCURRENTLY idx_embeddings_eudr_hnsw
ON vector_embeddings
USING hnsw (embedding vector_cosine_ops)
WHERE namespace = 'eudr';
```

---

## 7. Embedding Pipeline

### 7.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Embedding Pipeline Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                      Document Ingestion                           │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │       │
│  │  │  S3     │  │  API    │  │  Kafka  │  │  Manual │             │       │
│  │  │  Events │  │  Upload │  │  Stream │  │  Import │             │       │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘             │       │
│  │       └───────────┬┴───────────┬┴────────────┘                   │       │
│  └───────────────────┼────────────┼─────────────────────────────────┘       │
│                      ▼            │                                          │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                      Document Processor                           │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │       │
│  │  │  PDF       │  │  Text      │  │  HTML      │                 │       │
│  │  │  Extractor │  │  Cleaner   │  │  Parser    │                 │       │
│  │  └────────────┘  └────────────┘  └────────────┘                 │       │
│  └──────────────────────────┬───────────────────────────────────────┘       │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                         Chunking Service                          │       │
│  │  ┌────────────────────────────────────────────────────────────┐  │       │
│  │  │  Strategy: Semantic (preferred) | Fixed | Sliding Window   │  │       │
│  │  │  Chunk Size: 512 tokens | Overlap: 64 tokens               │  │       │
│  │  └────────────────────────────────────────────────────────────┘  │       │
│  └──────────────────────────┬───────────────────────────────────────┘       │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                       Embedding Service                           │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │       │
│  │  │  Model     │  │  Batch     │  │  Cache     │                 │       │
│  │  │  Inference │  │  Processor │  │  Layer     │                 │       │
│  │  │  (GPU/CPU) │  │  (1000/b)  │  │  (Redis)   │                 │       │
│  │  └────────────┘  └────────────┘  └────────────┘                 │       │
│  └──────────────────────────┬───────────────────────────────────────┘       │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                       pgvector Storage                            │       │
│  │  ┌────────────────────────────────────────────────────────────┐  │       │
│  │  │  Batch INSERT with COPY | Auto-vacuum | Index maintenance  │  │       │
│  │  └────────────────────────────────────────────────────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Embedding Service Specification

```python
# embedding_service.py - Core service interface

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    batch_size: int = 1000
    normalize: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 3600

@dataclass
class EmbeddingRequest:
    texts: List[str]
    namespace: str = "default"
    source_type: str = "document"
    source_id: Optional[str] = None
    metadata: Optional[dict] = None

@dataclass
class EmbeddingResult:
    embeddings: np.ndarray  # Shape: (n_texts, dimensions)
    model: str
    dimensions: int
    processing_time_ms: int

class EmbeddingService:
    """
    Production embedding service with:
    - Batch processing (1000 texts/batch)
    - GPU acceleration (optional)
    - Redis caching for repeated texts
    - Automatic retries with exponential backoff
    """

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate embeddings for texts."""
        pass

    async def embed_and_store(self, request: EmbeddingRequest) -> int:
        """Generate embeddings and store in pgvector."""
        pass

    async def search(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[dict]:
        """Semantic search using pgvector."""
        pass
```

### 7.3 Chunking Strategies

| Strategy | Use Case | Configuration |
|----------|----------|---------------|
| **Semantic** | Regulations, legal docs | Sentence boundaries, ~512 tokens |
| **Fixed** | Structured data, forms | Exact 512 tokens, 64 overlap |
| **Sliding Window** | Long narratives | 512 tokens, 128 overlap |
| **Hierarchical** | Complex documents | Parent-child chunks |

---

## 8. Query Optimization

### 8.1 Query Patterns

#### 8.1.1 Basic Similarity Search

```sql
-- Top-K similarity search with cosine distance
SELECT
    id,
    source_type,
    source_id,
    content_preview,
    1 - (embedding <=> $1::vector) AS similarity
FROM vector_embeddings
WHERE namespace = $2
ORDER BY embedding <=> $1::vector
LIMIT $3;
```

#### 8.1.2 Filtered Similarity Search

```sql
-- Filtered search with metadata conditions
SELECT
    id,
    source_type,
    source_id,
    content_preview,
    1 - (embedding <=> $1::vector) AS similarity
FROM vector_embeddings
WHERE
    namespace = $2
    AND source_type = $3
    AND metadata @> $4::jsonb
    AND 1 - (embedding <=> $1::vector) > $5  -- threshold
ORDER BY embedding <=> $1::vector
LIMIT $6;
```

#### 8.1.3 Hybrid Search (Vector + Full-Text)

```sql
-- Reciprocal Rank Fusion (RRF) hybrid search
WITH vector_results AS (
    SELECT
        id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS vector_rank
    FROM vector_embeddings
    WHERE namespace = $2
    ORDER BY embedding <=> $1::vector
    LIMIT 100
),
text_results AS (
    SELECT
        id,
        ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector(content_preview), plainto_tsquery($3)) DESC) AS text_rank
    FROM vector_embeddings
    WHERE
        namespace = $2
        AND to_tsvector(content_preview) @@ plainto_tsquery($3)
    LIMIT 100
),
rrf_scores AS (
    SELECT
        COALESCE(v.id, t.id) AS id,
        COALESCE(1.0 / (60 + v.vector_rank), 0) +
        COALESCE(1.0 / (60 + t.text_rank), 0) AS rrf_score
    FROM vector_results v
    FULL OUTER JOIN text_results t ON v.id = t.id
)
SELECT
    e.*,
    r.rrf_score
FROM rrf_scores r
JOIN vector_embeddings e ON e.id = r.id
ORDER BY r.rrf_score DESC
LIMIT $4;
```

### 8.2 Query Performance Tuning

```sql
-- Set search parameters for session
SET hnsw.ef_search = 100;  -- Higher = better recall, slower
SET work_mem = '256MB';     -- For large result sets

-- Analyze query plan
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT id, 1 - (embedding <=> $1::vector) AS similarity
FROM vector_embeddings
WHERE namespace = 'csrd'
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

### 8.3 Connection Pooling

```yaml
# PgBouncer configuration for vector queries
[databases]
greenlang_vectors = host=aurora-endpoint port=5432 dbname=greenlang

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 50
reserve_pool_size = 10
reserve_pool_timeout = 5
server_idle_timeout = 300
query_timeout = 30
```

---

## 9. Scaling Strategy

### 9.1 Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Scaling Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  READ SCALING:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Aurora Read Replicas                          │        │
│  │                                                                   │        │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐         │        │
│  │  │ Primary │   │ Reader  │   │ Reader  │   │ Reader  │         │        │
│  │  │ (Write) │   │   1     │   │   2     │   │   N     │         │        │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘         │        │
│  │       │              │             │             │               │        │
│  │       ▼              ▼             ▼             ▼               │        │
│  │  ┌─────────────────────────────────────────────────────────┐   │        │
│  │  │               PgBouncer Connection Pool                  │   │        │
│  │  │        (Read queries distributed to replicas)            │   │        │
│  │  └─────────────────────────────────────────────────────────┘   │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  WRITE SCALING:                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Batch Processing                              │        │
│  │                                                                   │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │        │
│  │  │   Batch 1   │  │   Batch 2   │  │   Batch N   │             │        │
│  │  │  1000 rows  │  │  1000 rows  │  │  1000 rows  │             │        │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │        │
│  │         │                │                │                      │        │
│  │         ▼                ▼                ▼                      │        │
│  │  ┌─────────────────────────────────────────────────────────┐   │        │
│  │  │              COPY Command (Bulk Insert)                  │   │        │
│  │  │           50,000 vectors/second throughput               │   │        │
│  │  └─────────────────────────────────────────────────────────┘   │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  STORAGE SCALING:                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │               Aurora Serverless v2 Auto-Scaling                  │        │
│  │                                                                   │        │
│  │  ┌───────────────────────────────────────────────────────────┐  │        │
│  │  │  Min: 2 ACU | Max: 128 ACU | Scale in 15s | Storage: Auto │  │        │
│  │  └───────────────────────────────────────────────────────────┘  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Scaling Thresholds

| Metric | Scale Up | Scale Down | Target |
|--------|----------|------------|--------|
| CPU Utilization | > 70% | < 30% | 50% |
| Memory Utilization | > 80% | < 40% | 60% |
| Query Latency P99 | > 100ms | < 50ms | 75ms |
| Connection Count | > 80% pool | < 30% pool | 50% |

### 9.3 Capacity Planning

| Vector Count | Storage | Memory (HNSW) | Instance Type |
|--------------|---------|---------------|---------------|
| 1M | 2 GB | 3 GB | db.r6g.large |
| 10M | 20 GB | 30 GB | db.r6g.xlarge |
| 100M | 200 GB | 300 GB | db.r6g.4xlarge |
| 1B | 2 TB | 3 TB | db.r6g.16xlarge |

---

## 10. Security Requirements

### 10.1 Network Security

```hcl
# Security group for pgvector access
resource "aws_security_group" "pgvector" {
  name_prefix = "greenlang-pgvector-"
  vpc_id      = var.vpc_id

  # Allow PostgreSQL from EKS
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.eks_security_group_id]
    description     = "PostgreSQL from EKS"
  }

  # Allow from embedding service
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.embedding_service_sg_id]
    description     = "PostgreSQL from embedding service"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }
}
```

### 10.2 Encryption

| Layer | Method | Key Management |
|-------|--------|----------------|
| At Rest | AES-256 | AWS KMS (CMK) |
| In Transit | TLS 1.3 | ACM certificates |
| Embeddings | Encrypted columns | Application-level |
| Backups | Encrypted snapshots | Same CMK |

### 10.3 Access Control

```sql
-- Role-based access for vector operations
CREATE ROLE vector_reader;
CREATE ROLE vector_writer;
CREATE ROLE vector_admin;

-- Reader permissions
GRANT USAGE ON SCHEMA public TO vector_reader;
GRANT SELECT ON vector_embeddings TO vector_reader;
GRANT SELECT ON embedding_collections TO vector_reader;

-- Writer permissions (includes reader)
GRANT vector_reader TO vector_writer;
GRANT INSERT, UPDATE ON vector_embeddings TO vector_writer;
GRANT INSERT, UPDATE ON embedding_jobs TO vector_writer;

-- Admin permissions (includes writer)
GRANT vector_writer TO vector_admin;
GRANT DELETE, TRUNCATE ON vector_embeddings TO vector_admin;
GRANT ALL ON embedding_collections TO vector_admin;
GRANT CREATE ON SCHEMA public TO vector_admin;

-- Application user
CREATE USER greenlang_app WITH PASSWORD '${var.db_password}';
GRANT vector_writer TO greenlang_app;

-- Admin user
CREATE USER greenlang_admin WITH PASSWORD '${var.admin_password}';
GRANT vector_admin TO greenlang_admin;
```

### 10.4 Audit Logging

```sql
-- Enable pgaudit for vector operations
CREATE EXTENSION IF NOT EXISTS pgaudit;

-- Configure audit logging
ALTER SYSTEM SET pgaudit.log = 'write, ddl';
ALTER SYSTEM SET pgaudit.log_catalog = off;
ALTER SYSTEM SET pgaudit.log_relation = on;
ALTER SYSTEM SET pgaudit.log_statement_once = on;

-- Audit trigger for vector changes
CREATE OR REPLACE FUNCTION audit_vector_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO vector_audit_log (
        operation,
        table_name,
        record_id,
        old_data,
        new_data,
        changed_by,
        changed_at
    ) VALUES (
        TG_OP,
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        CASE WHEN TG_OP = 'DELETE' THEN row_to_json(OLD) ELSE NULL END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) ELSE NULL END,
        current_user,
        NOW()
    );
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER vector_embeddings_audit
AFTER INSERT OR UPDATE OR DELETE ON vector_embeddings
FOR EACH ROW EXECUTE FUNCTION audit_vector_changes();
```

---

## 11. Monitoring and Observability

### 11.1 Metrics

#### 11.1.1 Vector-Specific Metrics

| Metric | Type | Description | Alert Threshold |
|--------|------|-------------|-----------------|
| `pgvector_embeddings_total` | Gauge | Total embeddings stored | N/A |
| `pgvector_search_latency_ms` | Histogram | Search query latency | P99 > 100ms |
| `pgvector_insert_rate` | Counter | Embeddings inserted/sec | < 100/s warning |
| `pgvector_index_size_bytes` | Gauge | HNSW index size | > 80% memory |
| `pgvector_search_recall` | Gauge | Search recall rate | < 0.95 |

#### 11.1.2 PostgreSQL Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `pg_stat_user_tables_n_tup_ins` | Rows inserted | N/A |
| `pg_stat_user_indexes_idx_scan` | Index scans | Low = missing index |
| `pg_stat_activity_count` | Active connections | > 80% pool |
| `pg_locks_count` | Lock contention | > 10 waiting |

### 11.2 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "pgvector Performance",
    "panels": [
      {
        "title": "Vector Search Latency",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(pgvector_search_latency_ms_bucket[5m]))",
            "legendFormat": "P99"
          },
          {
            "expr": "histogram_quantile(0.50, rate(pgvector_search_latency_ms_bucket[5m]))",
            "legendFormat": "P50"
          }
        ]
      },
      {
        "title": "Embeddings Count by Namespace",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(pgvector_embeddings_total) by (namespace)"
          }
        ]
      },
      {
        "title": "Index Memory Usage",
        "type": "gauge",
        "targets": [
          {
            "expr": "pgvector_index_size_bytes / pg_database_size_bytes * 100"
          }
        ]
      },
      {
        "title": "Search Throughput",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(pgvector_search_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### 11.3 Alerting Rules

```yaml
groups:
  - name: pgvector-alerts
    rules:
      - alert: PgvectorHighSearchLatency
        expr: histogram_quantile(0.99, rate(pgvector_search_latency_ms_bucket[5m])) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "pgvector search latency is high"
          description: "P99 search latency is {{ $value }}ms (threshold: 100ms)"

      - alert: PgvectorIndexMemoryHigh
        expr: pgvector_index_size_bytes / pg_settings_shared_buffers_bytes > 0.8
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "pgvector index using too much memory"
          description: "Index is using {{ $value | humanizePercentage }} of shared_buffers"

      - alert: PgvectorLowRecall
        expr: pgvector_search_recall < 0.95
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "pgvector search recall is degraded"
          description: "Search recall is {{ $value }} (threshold: 0.95)"

      - alert: PgvectorReplicationLag
        expr: pg_replication_lag_seconds > 30
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "pgvector replica lag is high"
          description: "Replication lag is {{ $value }}s (threshold: 30s)"
```

---

## 12. Backup and Recovery

### 12.1 Backup Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Backup Architecture                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  CONTINUOUS BACKUPS:                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                Aurora Automated Backups                          │        │
│  │  ┌───────────────────────────────────────────────────────────┐  │        │
│  │  │  Retention: 35 days | Point-in-time: 5-minute granularity │  │        │
│  │  │  Storage: S3 (encrypted) | Cross-region: Optional          │  │        │
│  │  └───────────────────────────────────────────────────────────┘  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  MANUAL SNAPSHOTS:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                 Scheduled Cluster Snapshots                      │        │
│  │  ┌───────────────────────────────────────────────────────────┐  │        │
│  │  │  Daily: 00:00 UTC | Weekly: Sunday 00:00 UTC               │  │        │
│  │  │  Monthly: 1st of month | Retention: 90 days                │  │        │
│  │  └───────────────────────────────────────────────────────────┘  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  VECTOR-SPECIFIC BACKUPS:                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                   Logical Exports                                │        │
│  │  ┌───────────────────────────────────────────────────────────┐  │        │
│  │  │  pg_dump: Daily (compressed) | Format: Custom              │  │        │
│  │  │  Tables: vector_embeddings, embedding_collections          │  │        │
│  │  │  Storage: S3 greenlang-backups/pgvector/                   │  │        │
│  │  └───────────────────────────────────────────────────────────┘  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Backup Configuration

```hcl
# Aurora backup configuration
resource "aws_rds_cluster" "greenlang" {
  # ... other config ...

  backup_retention_period      = 35
  preferred_backup_window      = "03:00-04:00"
  copy_tags_to_snapshot        = true
  deletion_protection          = true
  skip_final_snapshot          = false
  final_snapshot_identifier    = "greenlang-final-${formatdate("YYYY-MM-DD", timestamp())}"

  # Enable backtrack for point-in-time recovery
  backtrack_window = 72  # 72 hours
}

# Cross-region backup (disaster recovery)
resource "aws_rds_cluster_snapshot_copy" "dr" {
  source_db_cluster_snapshot_identifier = aws_db_cluster_snapshot.daily.id
  target_db_cluster_snapshot_identifier = "greenlang-dr-${formatdate("YYYY-MM-DD", timestamp())}"
  kms_key_id                            = var.dr_kms_key_arn

  # Copy to DR region
  provider = aws.dr_region
}
```

### 12.3 Recovery Procedures

#### 12.3.1 Point-in-Time Recovery

```bash
# Restore to specific point in time
aws rds restore-db-cluster-to-point-in-time \
  --source-db-cluster-identifier greenlang-prod \
  --db-cluster-identifier greenlang-recovery \
  --restore-to-time "2026-02-03T10:00:00Z" \
  --db-subnet-group-name greenlang-private \
  --vpc-security-group-ids sg-xxxxx \
  --kms-key-id alias/greenlang-rds

# Create instance in recovered cluster
aws rds create-db-instance \
  --db-instance-identifier greenlang-recovery-instance \
  --db-instance-class db.r6g.2xlarge \
  --db-cluster-identifier greenlang-recovery \
  --engine aurora-postgresql
```

#### 12.3.2 Snapshot Recovery

```bash
# Restore from snapshot
aws rds restore-db-cluster-from-snapshot \
  --db-cluster-identifier greenlang-restored \
  --snapshot-identifier greenlang-daily-2026-02-03 \
  --engine aurora-postgresql \
  --engine-version 15.4 \
  --db-subnet-group-name greenlang-private \
  --vpc-security-group-ids sg-xxxxx
```

### 12.4 Recovery Time Objectives

| Scenario | RTO | RPO | Procedure |
|----------|-----|-----|-----------|
| Instance Failure | 5 min | 0 | Aurora auto-failover |
| Data Corruption | 30 min | 5 min | Point-in-time restore |
| Full Cluster Loss | 2 hours | 1 hour | Snapshot restore |
| Regional Disaster | 4 hours | 24 hours | Cross-region restore |

---

## 13. Migration Strategy

### 13.1 Migration from Weaviate (Hybrid)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Weaviate to pgvector Migration                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  PHASE 1: PARALLEL OPERATION                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  - Deploy pgvector alongside Weaviate                            │        │
│  │  - Dual-write new embeddings to both                             │        │
│  │  - Read from Weaviate (primary)                                  │        │
│  │  Duration: 2 weeks                                               │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  PHASE 2: DATA MIGRATION                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  - Export Weaviate data in batches (10K vectors)                 │        │
│  │  - Transform to pgvector schema                                  │        │
│  │  - Import using COPY command                                     │        │
│  │  - Verify data integrity (checksums)                             │        │
│  │  Duration: 1-3 days (depending on volume)                        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  PHASE 3: TRAFFIC SHIFT                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  - Gradually shift read traffic: 10% → 50% → 100%                │        │
│  │  - Monitor latency and recall metrics                            │        │
│  │  - Rollback capability at each stage                             │        │
│  │  Duration: 1 week                                                │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  PHASE 4: DECOMMISSION                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  - Stop Weaviate writes                                          │        │
│  │  - Archive Weaviate data to S3                                   │        │
│  │  - Decommission Weaviate cluster                                 │        │
│  │  Duration: 1 week buffer                                         │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 13.2 Migration Scripts

```python
# weaviate_to_pgvector.py - Migration script

import weaviate
import psycopg
from pgvector.psycopg import register_vector
import numpy as np

BATCH_SIZE = 10000

async def migrate_collection(
    weaviate_client: weaviate.Client,
    pg_conn: psycopg.AsyncConnection,
    collection_name: str,
    namespace: str
):
    """Migrate a Weaviate collection to pgvector."""

    # Get total count
    result = weaviate_client.query.aggregate(collection_name).with_meta_count().do()
    total = result['data']['Aggregate'][collection_name][0]['meta']['count']

    offset = 0
    while offset < total:
        # Fetch batch from Weaviate
        batch = (
            weaviate_client.query
            .get(collection_name, ['content', 'source_id', 'metadata'])
            .with_additional(['vector', 'id'])
            .with_limit(BATCH_SIZE)
            .with_offset(offset)
            .do()
        )

        objects = batch['data']['Get'][collection_name]

        # Prepare for pgvector insert
        rows = []
        for obj in objects:
            rows.append((
                obj['_additional']['id'],
                'document',
                obj['source_id'],
                0,
                obj['content'][:500],
                np.array(obj['_additional']['vector']),
                'all-MiniLM-L6-v2',
                obj.get('metadata', {}),
                namespace
            ))

        # Bulk insert to pgvector
        async with pg_conn.cursor() as cur:
            await cur.executemany(
                """
                INSERT INTO vector_embeddings
                (id, source_type, source_id, chunk_index, content_preview,
                 embedding, embedding_model, metadata, namespace)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_type, source_id, chunk_index, embedding_model)
                DO NOTHING
                """,
                rows
            )

        offset += BATCH_SIZE
        print(f"Migrated {min(offset, total)}/{total} vectors")

    return total
```

---

## 14. Performance Requirements

### 14.1 SLOs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Search Latency P50 | < 20ms | Query execution time |
| Search Latency P99 | < 100ms | Query execution time |
| Insert Throughput | > 10,000/sec | Batch insert rate |
| Search Recall@10 | > 0.95 | Relevance testing |
| Availability | 99.9% | Uptime monitoring |

### 14.2 Load Testing

```yaml
# k6 load test configuration
scenarios:
  vector_search:
    executor: ramping-vus
    startVUs: 10
    stages:
      - duration: 2m
        target: 100
      - duration: 5m
        target: 100
      - duration: 2m
        target: 200
      - duration: 5m
        target: 200
      - duration: 2m
        target: 0

  batch_insert:
    executor: constant-arrival-rate
    rate: 1000
    timeUnit: 1s
    duration: 10m
    preAllocatedVUs: 50

thresholds:
  http_req_duration:
    - p(50) < 20
    - p(99) < 100
  http_req_failed:
    - rate < 0.01
```

### 14.3 Benchmark Results (Expected)

| Operation | Vectors | Dimensions | Latency | Throughput |
|-----------|---------|------------|---------|------------|
| Insert (batch) | 1M | 384 | 20s | 50K/s |
| Search top-10 | 1M | 384 | 5ms | 2000 QPS |
| Search top-10 | 10M | 384 | 15ms | 1000 QPS |
| Search top-10 | 100M | 384 | 50ms | 500 QPS |
| Hybrid search | 10M | 384 | 30ms | 500 QPS |

---

## 15. Cost Analysis

### 15.1 Infrastructure Costs

| Component | Development | Staging | Production |
|-----------|-------------|---------|------------|
| Aurora Primary | $200/mo | $400/mo | $800/mo |
| Aurora Readers (2x) | - | $400/mo | $800/mo |
| Storage (per GB) | $0.10 | $0.10 | $0.10 |
| Backup Storage | $20/mo | $50/mo | $100/mo |
| Data Transfer | $10/mo | $50/mo | $200/mo |
| **Total** | **$230/mo** | **$900/mo** | **$1,900/mo** |

### 15.2 Cost Comparison

| Solution | 100M Vectors/mo | Features |
|----------|-----------------|----------|
| **pgvector (Aurora)** | $1,900 | ACID, unified DB, SQL |
| Weaviate (Dedicated) | $3,500 | Native vector, GraphQL |
| Pinecone (Serverless) | $4,000 | Managed, serverless |
| Milvus (Self-hosted) | $2,500 | Open source, complex |

### 15.3 Cost Optimization

1. **Reserved Instances**: 40% savings with 1-year commitment
2. **Right-sizing**: Use Aurora Serverless v2 for variable workloads
3. **Read Replicas**: Scale reads without scaling writes
4. **Compression**: Use halfvec (16-bit) for 50% storage reduction
5. **Lifecycle**: Archive old embeddings to S3

---

## 16. Implementation Roadmap

### 16.1 Phase 1: Foundation (Week 1-2)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Deploy pgvector extension | Platform | Aurora cluster with pgvector |
| Create schema | Platform | Tables, indexes, roles |
| Configure security | Security | Encryption, IAM, audit |
| Set up monitoring | SRE | Dashboards, alerts |

### 16.2 Phase 2: Integration (Week 3-4)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Embedding service | ML Eng | Python SDK |
| RAG integration | ML Eng | Updated RAG engine |
| API endpoints | Backend | Vector search API |
| Load testing | QA | Performance report |

### 16.3 Phase 3: Migration (Week 5-6)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Weaviate export | Platform | Migrated data |
| Data validation | QA | Integrity report |
| Traffic shift | SRE | Production cutover |
| Documentation | Tech Writer | Runbooks, guides |

### 16.4 Milestones

| Milestone | Date | Criteria |
|-----------|------|----------|
| M1: Infrastructure Ready | Week 2 | pgvector deployed, schema created |
| M2: Integration Complete | Week 4 | RAG using pgvector in staging |
| M3: Production Ready | Week 6 | 100% traffic on pgvector |
| M4: Weaviate Decommissioned | Week 8 | Full migration complete |

---

## 17. Success Criteria

### 17.1 Technical Success

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Search Latency | P99 < 100ms | Prometheus metrics |
| Availability | > 99.9% | Uptime monitoring |
| Data Integrity | 100% | Checksums, validation |
| Recall Rate | > 0.95 | Benchmark tests |

### 17.2 Business Success

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Cost Reduction | 40% | AWS billing |
| Operational Simplicity | Single DB | Architecture review |
| Feature Velocity | +20% | Sprint metrics |
| Regulatory Compliance | 100% | Audit report |

### 17.3 Acceptance Tests

```gherkin
Feature: pgvector Vector Search

  Scenario: Semantic search returns relevant results
    Given 1 million embeddings are stored
    And a query embedding is generated
    When I search for top-10 similar vectors
    Then the response time is less than 100ms
    And the recall rate is above 0.95

  Scenario: Hybrid search combines vector and text
    Given documents with embeddings and full text
    When I perform a hybrid search
    Then results are ranked by RRF score
    And response time is less than 150ms

  Scenario: Batch insert handles high throughput
    Given 10,000 embeddings to insert
    When I perform a batch insert
    Then all embeddings are stored
    And throughput exceeds 10,000/second
```

---

## 18. Appendices

### 18.1 pgvector Extension Reference

```sql
-- Available distance operators
-- <-> : L2 (Euclidean) distance
-- <#> : Inner product (negative)
-- <=> : Cosine distance

-- Example usage
SELECT embedding <=> '[0.1, 0.2, ...]'::vector AS distance;

-- Index operator classes
-- vector_l2_ops    : L2 distance
-- vector_ip_ops    : Inner product
-- vector_cosine_ops: Cosine distance
```

### 18.2 Recommended PostgreSQL Settings

```sql
-- postgresql.conf for pgvector workloads
shared_buffers = '16GB'              -- 25% of RAM
effective_cache_size = '48GB'        -- 75% of RAM
maintenance_work_mem = '2GB'         -- For index builds
work_mem = '256MB'                   -- Per-query memory
max_parallel_workers_per_gather = 4  -- Parallel queries
random_page_cost = 1.1               -- SSD storage

-- pgvector specific
hnsw.ef_search = 100                 -- Query-time search width
```

### 18.3 Glossary

| Term | Definition |
|------|------------|
| **HNSW** | Hierarchical Navigable Small World - graph-based index |
| **IVFFlat** | Inverted File with Flat quantization - partition-based index |
| **Cosine Distance** | 1 - cosine_similarity; measures angle between vectors |
| **Recall@K** | Fraction of true nearest neighbors in top-K results |
| **RRF** | Reciprocal Rank Fusion - method to combine multiple rankings |
| **ACU** | Aurora Capacity Unit - serverless compute unit |

### 18.4 References

1. [pgvector Documentation](https://github.com/pgvector/pgvector)
2. [Aurora PostgreSQL Best Practices](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/Aurora.BestPractices.html)
3. [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
4. [Sentence Transformers](https://www.sbert.net/)

---

## Document Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Platform Engineering Lead | | | |
| ML Engineering Lead | | | |
| Security Lead | | | |
| VP of Engineering | | | |

---

**End of PRD-INFRA-005-Vector-DB-pgvector.md**
