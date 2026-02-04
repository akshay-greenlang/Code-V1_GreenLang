################################################################################
# Aurora PostgreSQL Parameter Groups
# Custom parameter groups optimized for TimescaleDB extension
################################################################################

################################################################################
# Cluster Parameter Group
################################################################################

resource "aws_rds_cluster_parameter_group" "aurora_timescaledb" {
  name        = "${local.name_prefix}-aurora-cluster-pg"
  family      = "aurora-postgresql15"
  description = "Aurora PostgreSQL cluster parameter group with TimescaleDB support"

  # ============================================================================
  # TimescaleDB Configuration
  # ============================================================================

  # Load TimescaleDB + pgvector extensions
  parameter {
    name         = "shared_preload_libraries"
    value        = "pg_stat_statements,timescaledb,pgaudit"
    apply_method = "pending-reboot"
  }

  # ============================================================================
  # pgvector Configuration (INFRA-005)
  # ============================================================================

  # HNSW ef_search: query-time search width (higher = better recall, slower)
  parameter {
    name         = "hnsw.ef_search"
    value        = var.pgvector_hnsw_ef_search
    apply_method = "immediate"
  }

  # pgaudit logging for vector operations
  parameter {
    name         = "pgaudit.log"
    value        = var.pgvector_audit_log_level
    apply_method = "immediate"
  }

  parameter {
    name         = "pgaudit.log_relation"
    value        = "on"
    apply_method = "immediate"
  }

  parameter {
    name         = "pgaudit.log_statement_once"
    value        = "on"
    apply_method = "immediate"
  }

  # TimescaleDB max background workers (for continuous aggregates, compression, etc.)
  parameter {
    name         = "timescaledb.max_background_workers"
    value        = var.timescaledb_max_background_workers
    apply_method = "pending-reboot"
  }

  # TimescaleDB telemetry (off for production privacy)
  parameter {
    name         = "timescaledb.telemetry_level"
    value        = var.timescaledb_telemetry_level
    apply_method = "immediate"
  }

  # ============================================================================
  # Replication Configuration (INFRA-002)
  # ============================================================================

  # Synchronous commit for zero-data-loss replication
  parameter {
    name         = "synchronous_commit"
    value        = var.synchronous_commit
    apply_method = "immediate"
  }

  # ============================================================================
  # Connection Limits
  # ============================================================================

  # Maximum number of concurrent connections
  parameter {
    name         = "max_connections"
    value        = var.max_connections
    apply_method = "pending-reboot"
  }

  # Idle transaction session timeout (prevent connection leaks)
  parameter {
    name         = "idle_in_transaction_session_timeout"
    value        = var.idle_in_transaction_session_timeout
    apply_method = "immediate"
  }

  # Statement timeout (0 = no timeout)
  parameter {
    name         = "statement_timeout"
    value        = var.statement_timeout
    apply_method = "immediate"
  }

  # ============================================================================
  # Memory Settings
  # ============================================================================

  # Work memory for complex operations
  parameter {
    name         = "work_mem"
    value        = "262144"  # 256 MB
    apply_method = "immediate"
  }

  # Maintenance work memory for VACUUM, CREATE INDEX, etc.
  parameter {
    name         = "maintenance_work_mem"
    value        = "524288"  # 512 MB
    apply_method = "immediate"
  }

  # Effective cache size (hint for query planner)
  parameter {
    name         = "effective_cache_size"
    value        = "{DBInstanceClassMemory*3/4}"
    apply_method = "immediate"
  }

  # Shared buffers (Aurora manages this, but set a reasonable default)
  parameter {
    name         = "shared_buffers"
    value        = "{DBInstanceClassMemory/4}"
    apply_method = "pending-reboot"
  }

  # ============================================================================
  # WAL Configuration
  # ============================================================================

  # WAL level for replication
  parameter {
    name         = "wal_level"
    value        = "logical"
    apply_method = "pending-reboot"
  }

  # Maximum WAL senders (for replication)
  parameter {
    name         = "max_wal_senders"
    value        = "10"
    apply_method = "pending-reboot"
  }

  # Maximum replication slots
  parameter {
    name         = "max_replication_slots"
    value        = "10"
    apply_method = "pending-reboot"
  }

  # WAL keep size (for replication lag tolerance)
  parameter {
    name         = "wal_keep_size"
    value        = "2048"  # 2 GB
    apply_method = "immediate"
  }

  # ============================================================================
  # Logging Settings
  # ============================================================================

  # Log destination (stderr for CloudWatch)
  parameter {
    name         = "log_destination"
    value        = "stderr"
    apply_method = "immediate"
  }

  # Log checkpoints
  parameter {
    name         = "log_checkpoints"
    value        = "1"
    apply_method = "immediate"
  }

  # Log connections
  parameter {
    name         = "log_connections"
    value        = "1"
    apply_method = "immediate"
  }

  # Log disconnections
  parameter {
    name         = "log_disconnections"
    value        = "1"
    apply_method = "immediate"
  }

  # Log lock waits
  parameter {
    name         = "log_lock_waits"
    value        = "1"
    apply_method = "immediate"
  }

  # Minimum log duration for statements (1000ms = 1 second)
  parameter {
    name         = "log_min_duration_statement"
    value        = "1000"
    apply_method = "immediate"
  }

  # Log statements (none, ddl, mod, all)
  parameter {
    name         = "log_statement"
    value        = "ddl"
    apply_method = "immediate"
  }

  # Log temp files larger than this size (KB)
  parameter {
    name         = "log_temp_files"
    value        = "10240"  # 10 MB
    apply_method = "immediate"
  }

  # ============================================================================
  # Query Planner Settings
  # ============================================================================

  # Random page cost (lower for SSD storage)
  parameter {
    name         = "random_page_cost"
    value        = "1.1"
    apply_method = "immediate"
  }

  # Effective IO concurrency
  parameter {
    name         = "effective_io_concurrency"
    value        = "200"
    apply_method = "immediate"
  }

  # Default statistics target
  parameter {
    name         = "default_statistics_target"
    value        = "100"
    apply_method = "immediate"
  }

  # ============================================================================
  # Background Workers
  # ============================================================================

  # Maximum worker processes
  parameter {
    name         = "max_worker_processes"
    value        = "16"
    apply_method = "pending-reboot"
  }

  # Maximum parallel workers per gather
  parameter {
    name         = "max_parallel_workers_per_gather"
    value        = "4"
    apply_method = "immediate"
  }

  # Maximum parallel workers
  parameter {
    name         = "max_parallel_workers"
    value        = "8"
    apply_method = "immediate"
  }

  # Maximum parallel maintenance workers
  parameter {
    name         = "max_parallel_maintenance_workers"
    value        = "4"
    apply_method = "immediate"
  }

  # ============================================================================
  # Autovacuum Settings
  # ============================================================================

  # Autovacuum max workers
  parameter {
    name         = "autovacuum_max_workers"
    value        = "5"
    apply_method = "pending-reboot"
  }

  # Autovacuum naptime
  parameter {
    name         = "autovacuum_naptime"
    value        = "30"  # 30 seconds
    apply_method = "immediate"
  }

  # Autovacuum vacuum threshold
  parameter {
    name         = "autovacuum_vacuum_threshold"
    value        = "50"
    apply_method = "immediate"
  }

  # Autovacuum analyze threshold
  parameter {
    name         = "autovacuum_analyze_threshold"
    value        = "50"
    apply_method = "immediate"
  }

  # Autovacuum vacuum scale factor
  parameter {
    name         = "autovacuum_vacuum_scale_factor"
    value        = "0.1"
    apply_method = "immediate"
  }

  # Autovacuum analyze scale factor
  parameter {
    name         = "autovacuum_analyze_scale_factor"
    value        = "0.05"
    apply_method = "immediate"
  }

  # ============================================================================
  # pg_stat_statements Configuration
  # ============================================================================

  # Track all statements
  parameter {
    name         = "pg_stat_statements.track"
    value        = "all"
    apply_method = "immediate"
  }

  # Maximum tracked statements
  parameter {
    name         = "pg_stat_statements.max"
    value        = "10000"
    apply_method = "pending-reboot"
  }

  # ============================================================================
  # SSL Configuration
  # ============================================================================

  # Force SSL connections
  parameter {
    name         = "rds.force_ssl"
    value        = "1"
    apply_method = "immediate"
  }

  # ============================================================================
  # Additional Custom Parameters
  # ============================================================================

  dynamic "parameter" {
    for_each = var.cluster_parameters
    content {
      name         = parameter.value.name
      value        = parameter.value.value
      apply_method = parameter.value.apply_method
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-cluster-pg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

################################################################################
# Instance Parameter Group
################################################################################

resource "aws_db_parameter_group" "aurora_timescaledb" {
  name        = "${local.name_prefix}-aurora-instance-pg"
  family      = "aurora-postgresql15"
  description = "Aurora PostgreSQL instance parameter group with TimescaleDB support"

  # ============================================================================
  # Instance-Level Memory Settings
  # ============================================================================

  # Huge pages (try to use if available)
  parameter {
    name         = "huge_pages"
    value        = "on"
    apply_method = "pending-reboot"
  }

  # ============================================================================
  # Instance-Level Logging Settings
  # ============================================================================

  # Log line prefix for better log parsing
  parameter {
    name         = "log_line_prefix"
    value        = "%t:%r:%u@%d:[%p]:"
    apply_method = "immediate"
  }

  # Log error verbosity
  parameter {
    name         = "log_error_verbosity"
    value        = "default"
    apply_method = "immediate"
  }

  # ============================================================================
  # Instance-Level Performance Settings
  # ============================================================================

  # Enable JIT compilation
  parameter {
    name         = "jit"
    value        = "on"
    apply_method = "immediate"
  }

  # JIT cost threshold
  parameter {
    name         = "jit_above_cost"
    value        = "100000"
    apply_method = "immediate"
  }

  # ============================================================================
  # Additional Custom Parameters
  # ============================================================================

  dynamic "parameter" {
    for_each = var.instance_parameters
    content {
      name         = parameter.value.name
      value        = parameter.value.value
      apply_method = parameter.value.apply_method
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-instance-pg"
  })

  lifecycle {
    create_before_destroy = true
  }
}
