# Pipeline Checkpointing Architecture

## Executive Summary

GreenLang's Pipeline Checkpointing System provides enterprise-grade resilience for long-running data pipelines, enabling automatic recovery from failures and efficient resource utilization through pause/resume capabilities.

## Key Features

### 1. **Multi-Strategy Storage Support**
- **File-based**: JSON checkpoints for development/testing
- **PostgreSQL**: ACID-compliant storage for production
- **Redis**: High-performance in-memory checkpoints
- **SQLite**: Lightweight embedded database option
- **Memory**: Testing and development scenarios

### 2. **Automatic Failure Recovery**
- Detect interrupted pipelines automatically
- Resume from last successful checkpoint
- Skip already completed stages
- Preserve intermediate results and provenance

### 3. **Checkpoint Management**
- Configurable retention policies
- Automatic cleanup of old checkpoints
- Export/import for backup and migration
- Checkpoint validation and repair tools

### 4. **Production-Ready Features**
- SHA-256 integrity verification
- Provenance tracking for audit trails
- Performance monitoring and statistics
- Visual debugging tools

## Architecture Overview

```mermaid
graph TB
    subgraph "Pipeline Execution Layer"
        P[Pipeline] --> |Execute| S1[Stage 1]
        S1 --> |Checkpoint| CP1[Checkpoint 1]
        CP1 --> S2[Stage 2]
        S2 --> |Checkpoint| CP2[Checkpoint 2]
        CP2 --> S3[Stage 3]
        S3 --> |Checkpoint| CP3[Checkpoint 3]
    end

    subgraph "Checkpoint Manager"
        CM[CheckpointManager] --> |Strategy Pattern| CS[Checkpoint Storage]
        CS --> FS[File Storage]
        CS --> DS[Database Storage]
        CS --> RS[Redis Storage]
        CS --> MS[Memory Storage]
    end

    subgraph "Recovery System"
        RF[Resume Flow] --> |Load| LC[Latest Checkpoint]
        LC --> |Restore| RS2[Restore State]
        RS2 --> |Continue| NP[Next Pipeline Stage]
    end

    P --> CM
    CP1 --> CS
    CP2 --> CS
    CP3 --> CS
```

## Component Architecture

### CheckpointManager
```python
CheckpointManager
├── Strategy Selection
│   ├── File (JSON)
│   ├── Database (PostgreSQL)
│   ├── Redis
│   └── Memory
├── Checkpoint Operations
│   ├── create_checkpoint()
│   ├── resume_pipeline()
│   ├── update_status()
│   └── cleanup_old()
└── Management Tools
    ├── export_checkpoint()
    ├── import_checkpoint()
    └── get_statistics()
```

### Checkpoint Storage Hierarchy
```python
CheckpointStorage (Abstract)
├── FileCheckpointStorage
│   ├── JSON serialization
│   ├── Atomic file operations
│   └── Directory-based organization
├── DatabaseCheckpointStorage
│   ├── PostgreSQL tables
│   ├── JSONB columns
│   └── Index optimization
├── RedisCheckpointStorage
│   ├── Binary serialization
│   ├── TTL-based expiration
│   └── Pipeline indexing
└── MemoryCheckpointStorage
    ├── In-process storage
    └── Testing scenarios
```

## Data Flow

### 1. Checkpoint Creation Flow
```mermaid
sequenceDiagram
    participant Pipeline
    participant Agent
    participant CheckpointManager
    participant Storage

    Pipeline->>Agent: Execute Stage
    Agent->>Pipeline: Return Output
    Pipeline->>CheckpointManager: Create Checkpoint
    CheckpointManager->>CheckpointManager: Calculate SHA-256
    CheckpointManager->>Storage: Save Checkpoint
    Storage->>CheckpointManager: Confirm Save
    CheckpointManager->>Pipeline: Return Checkpoint ID
```

### 2. Resume Flow
```mermaid
sequenceDiagram
    participant Pipeline
    participant CheckpointManager
    participant Storage
    participant Agent

    Pipeline->>CheckpointManager: Resume Request
    CheckpointManager->>Storage: Load Latest Checkpoint
    Storage->>CheckpointManager: Return Checkpoint
    CheckpointManager->>CheckpointManager: Validate Integrity
    CheckpointManager->>Pipeline: Return Resume Context
    Pipeline->>Pipeline: Restore State
    Pipeline->>Agent: Continue from Next Stage
```

## Checkpoint Data Model

```python
PipelineCheckpoint:
    metadata: CheckpointMetadata
        pipeline_id: str
        stage_name: str
        stage_index: int
        timestamp: datetime
        status: CheckpointStatus
        checksum: str (SHA-256)
        data_size: int
        error_message: Optional[str]
        resume_count: int
        parent_checkpoint_id: Optional[str]

    state_data: Dict[str, Any]
        inputs: Dict
        pipeline_config: Dict

    completed_stages: List[str]
    pending_stages: List[str]
    agent_outputs: Dict[str, Any]
    provenance_hashes: Dict[str, str]
```

## Usage Examples

### Basic Pipeline with Checkpointing

```python
from greenlang.sdk.pipeline import Pipeline

# Create pipeline with checkpointing enabled
pipeline = Pipeline(
    name="etl_pipeline",
    checkpoint_enabled=True,
    checkpoint_strategy="file",
    checkpoint_config={
        "base_path": "/var/greenlang/checkpoints"
    },
    auto_resume=True,  # Auto-resume from checkpoint
    steps=[
        {"name": "extract", "agent": "ExtractAgent"},
        {"name": "transform", "agent": "TransformAgent"},
        {"name": "load", "agent": "LoadAgent"}
    ]
)

# Execute with automatic checkpointing
results = pipeline.execute()

# If pipeline fails, next execution will auto-resume
results = pipeline.execute(resume=True)
```

### Production Configuration with PostgreSQL

```python
pipeline = Pipeline(
    name="compliance_pipeline",
    checkpoint_enabled=True,
    checkpoint_strategy="database",
    checkpoint_config={
        "connection_string": "postgresql://user:pass@host/db"
    },
    checkpoint_after_each_step=True,
    steps=[...]
)
```

### High-Performance with Redis

```python
pipeline = Pipeline(
    name="realtime_pipeline",
    checkpoint_enabled=True,
    checkpoint_strategy="redis",
    checkpoint_config={
        "host": "redis.internal",
        "port": 6379,
        "ttl_seconds": 3600  # 1 hour TTL
    },
    steps=[...]
)
```

## Debugging Tools

### Checkpoint Inspector
```bash
python -m greenlang.pipeline.checkpoint_debugger inspect \
    --checkpoint-id pipeline_20240101_stage2_120000
```

### Pipeline Analysis
```bash
python -m greenlang.pipeline.checkpoint_debugger analyze \
    --pipeline-id etl_pipeline_abc123_20240101
```

### Checkpoint Visualization
```bash
python -m greenlang.pipeline.checkpoint_debugger visualize \
    --pipeline-id etl_pipeline \
    --output checkpoint_graph.png
```

### Recovery Report
```bash
python -m greenlang.pipeline.checkpoint_debugger report \
    --pipeline-id failed_pipeline
```

## Performance Characteristics

| Storage Strategy | Write Speed | Read Speed | Persistence | Scalability | Use Case |
|-----------------|-------------|------------|-------------|-------------|----------|
| File | Medium | Medium | Yes | Limited | Development, Small scale |
| PostgreSQL | Medium | Fast | Yes | High | Production, Enterprise |
| Redis | Very Fast | Very Fast | Configurable | High | Real-time, Distributed |
| SQLite | Fast | Fast | Yes | Medium | Embedded, Edge |
| Memory | Instant | Instant | No | Limited | Testing |

## Best Practices

### 1. **Storage Strategy Selection**
```python
# Development
checkpoint_strategy="file"

# Production with compliance requirements
checkpoint_strategy="database"

# High-performance distributed systems
checkpoint_strategy="redis"

# Edge computing
checkpoint_strategy="sqlite"
```

### 2. **Retention Policies**
```python
manager = CheckpointManager(...)
# Clean up checkpoints older than 7 days
manager.auto_cleanup(retention_days=7)
```

### 3. **Error Handling**
```python
try:
    results = pipeline.execute()
except Exception as e:
    # Checkpoint automatically created on failure
    # Can resume later
    recovery_info = pipeline.get_checkpoint_history()
```

### 4. **Monitoring**
```python
# Get checkpoint statistics
stats = manager.get_statistics()
print(f"Total checkpoints: {stats['total_checkpoints']}")
print(f"Resume success rate: {stats['resume_success_rate']}")
```

## Security Considerations

### 1. **Data Encryption**
- Checkpoints can contain sensitive data
- Use encrypted file systems for file storage
- Enable SSL/TLS for database connections
- Use Redis AUTH and SSL for Redis storage

### 2. **Access Control**
- Implement role-based access to checkpoint storage
- Audit checkpoint access and modifications
- Use separate storage for different security levels

### 3. **Data Retention**
- Comply with data retention regulations
- Implement automatic cleanup policies
- Securely delete expired checkpoints

## Migration Guide

### Migrating from No Checkpointing
```python
# Before
pipeline = Pipeline(name="my_pipeline", steps=[...])

# After
pipeline = Pipeline(
    name="my_pipeline",
    checkpoint_enabled=True,  # Enable checkpointing
    checkpoint_strategy="file",  # Start with file storage
    auto_resume=True,  # Enable auto-resume
    steps=[...]
)
```

### Migrating Storage Strategies
```python
# Export from file storage
file_manager = CheckpointManager(strategy="file")
file_manager.export_checkpoint(checkpoint_id, "backup.json")

# Import to database storage
db_manager = CheckpointManager(strategy="database")
db_manager.import_checkpoint("backup.json")
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Checkpoints not created | Storage permissions | Check file/database permissions |
| Resume fails | Corrupted checkpoint | Use repair tool or find previous valid checkpoint |
| High storage usage | No cleanup policy | Implement auto_cleanup with retention |
| Slow checkpoint creation | Large state data | Use Redis or optimize state size |
| Lost checkpoints | Storage failure | Implement backup strategy |

## Future Enhancements

### Planned Features
- **Checkpoint Compression**: Reduce storage requirements
- **Encrypted Checkpoints**: Built-in encryption support
- **Cloud Storage**: S3, Azure Blob, GCS backends
- **Checkpoint Streaming**: Real-time checkpoint replication
- **Smart Recovery**: ML-based optimal recovery point selection
- **Checkpoint Versioning**: Support for schema evolution

## Conclusion

GreenLang's Pipeline Checkpointing System provides:
- **Reliability**: Automatic recovery from failures
- **Flexibility**: Multiple storage strategies
- **Performance**: Optimized for various workloads
- **Observability**: Comprehensive debugging tools
- **Security**: Enterprise-grade data protection

The system enables building resilient, production-grade data pipelines that can handle failures gracefully and maintain data integrity throughout the processing lifecycle.