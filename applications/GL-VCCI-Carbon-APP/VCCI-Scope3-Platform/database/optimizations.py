# -*- coding: utf-8 -*-
"""
Database Query Optimization Module
GL-VCCI Scope 3 Platform - Performance Optimization

This module provides comprehensive database query optimization capabilities including:
- Composite indexes for frequently queried columns
- Query performance analysis
- N+1 query detection and optimization
- Query result caching
- Database statistics collection

Performance Targets:
- Query latency P95: <50ms
- Query latency P99: <100ms
- Complex joins: <200ms
- Full table scans eliminated

Version: 1.0.0
Team: Performance Optimization Team
Date: 2025-11-09
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import sqlalchemy as sa
from sqlalchemy import text, Index, MetaData, Table
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.sql import Select
from greenlang.determinism import FinancialDecimal

logger = logging.getLogger(__name__)


# ============================================================================
# INDEX DEFINITIONS
# ============================================================================

# Composite indexes for emissions table
EMISSIONS_INDEXES = [
    {
        "name": "idx_emissions_composite",
        "table": "emissions",
        "columns": ["supplier_id", "scope3_category", "transaction_date"],
        "description": "Primary composite index for common queries"
    },
    {
        "name": "idx_emissions_supplier",
        "table": "emissions",
        "columns": ["supplier_id"],
        "description": "Single-column index for supplier lookups"
    },
    {
        "name": "idx_emissions_category",
        "table": "emissions",
        "columns": ["scope3_category"],
        "description": "Single-column index for category filtering"
    },
    {
        "name": "idx_emissions_date",
        "table": "emissions",
        "columns": ["transaction_date"],
        "description": "Single-column index for date range queries"
    },
    {
        "name": "idx_emissions_reporting",
        "table": "emissions",
        "columns": ["transaction_date", "scope3_category"],
        "include": ["emissions_tco2e", "dqi_score"],
        "description": "Covering index for reporting queries"
    },
    {
        "name": "idx_emissions_tenant",
        "table": "emissions",
        "columns": ["tenant_id", "transaction_date"],
        "description": "Multi-tenant isolation index"
    }
]

# Indexes for suppliers table
SUPPLIERS_INDEXES = [
    {
        "name": "idx_suppliers_name",
        "table": "suppliers",
        "columns": ["supplier_name"],
        "description": "Supplier name lookup index"
    },
    {
        "name": "idx_suppliers_tenant",
        "table": "suppliers",
        "columns": ["tenant_id"],
        "description": "Multi-tenant isolation for suppliers"
    },
    {
        "name": "idx_suppliers_duns",
        "table": "suppliers",
        "columns": ["duns_number"],
        "where": "duns_number IS NOT NULL",
        "description": "DUNS number lookup (partial index)"
    },
    {
        "name": "idx_suppliers_lei",
        "table": "suppliers",
        "columns": ["lei_code"],
        "where": "lei_code IS NOT NULL",
        "description": "LEI code lookup (partial index)"
    }
]

# Indexes for calculations table
CALCULATIONS_INDEXES = [
    {
        "name": "idx_calculations_batch",
        "table": "calculations",
        "columns": ["batch_id", "created_at"],
        "description": "Batch processing lookup"
    },
    {
        "name": "idx_calculations_status",
        "table": "calculations",
        "columns": ["status", "created_at"],
        "description": "Status-based queries"
    },
    {
        "name": "idx_calculations_tenant",
        "table": "calculations",
        "columns": ["tenant_id", "created_at"],
        "description": "Multi-tenant calculations"
    }
]

# All indexes
ALL_INDEXES = EMISSIONS_INDEXES + SUPPLIERS_INDEXES + CALCULATIONS_INDEXES


# ============================================================================
# QUERY PERFORMANCE ANALYZER
# ============================================================================

@dataclass
class QueryAnalysis:
    """Query performance analysis result"""
    query_hash: str
    query_text: str
    execution_time_ms: float
    plan_cost: Optional[float]
    rows_examined: Optional[int]
    rows_returned: Optional[int]
    uses_index: bool
    full_table_scan: bool
    recommendations: List[str]
    timestamp: datetime


class QueryAnalyzer:
    """
    Analyzes database query performance using EXPLAIN ANALYZE.

    Features:
    - Query plan analysis
    - Index usage detection
    - Full table scan detection
    - Performance recommendations
    - Slow query detection
    """

    def __init__(self, slow_query_threshold_ms: float = 1000):
        """
        Initialize query analyzer.

        Args:
            slow_query_threshold_ms: Threshold for slow query detection (default: 1000ms)
        """
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.query_history: List[QueryAnalysis] = []

    async def analyze_query(
        self,
        session: AsyncSession,
        query: Select,
        include_execution: bool = True
    ) -> QueryAnalysis:
        """
        Analyze query performance using EXPLAIN ANALYZE.

        Args:
            session: Database session
            query: SQLAlchemy query to analyze
            include_execution: Whether to execute query (ANALYZE) or just plan (EXPLAIN)

        Returns:
            QueryAnalysis with performance metrics and recommendations
        """
        import hashlib

        # Convert query to text
        query_text = str(query.compile(compile_kwargs={"literal_binds": True}))
        query_hash = hashlib.md5(query_text.encode()).hexdigest()

        start_time = time.time()

        try:
            # Get query plan
            if include_execution:
                explain_query = f"EXPLAIN ANALYZE {query_text}"
            else:
                explain_query = f"EXPLAIN {query_text}"

            result = await session.execute(text(explain_query))
            plan_lines = [row[0] for row in result.fetchall()]

            execution_time_ms = (time.time() - start_time) * 1000

            # Parse plan
            uses_index = any("Index Scan" in line for line in plan_lines)
            full_table_scan = any("Seq Scan" in line for line in plan_lines)

            # Extract metrics
            plan_cost = None
            rows_examined = None
            rows_returned = None

            for line in plan_lines:
                if "cost=" in line:
                    try:
                        cost_part = line.split("cost=")[1].split()[0]
                        plan_cost = FinancialDecimal.from_string(cost_part.split("..")[-1])
                    except:
                        pass

                if "rows=" in line:
                    try:
                        rows_part = line.split("rows=")[1].split()[0]
                        rows_examined = int(rows_part)
                    except:
                        pass

            # Generate recommendations
            recommendations = []

            if full_table_scan and not uses_index:
                recommendations.append(
                    "CRITICAL: Query uses full table scan. Add appropriate indexes."
                )

            if execution_time_ms > self.slow_query_threshold_ms:
                recommendations.append(
                    f"SLOW QUERY: Execution time {execution_time_ms:.2f}ms exceeds "
                    f"threshold {self.slow_query_threshold_ms}ms"
                )

            if plan_cost and plan_cost > 1000:
                recommendations.append(
                    f"HIGH COST: Query plan cost {plan_cost:.2f} is high. "
                    "Consider query optimization."
                )

            if rows_examined and rows_returned and rows_examined > rows_returned * 10:
                recommendations.append(
                    f"INEFFICIENT: Query examines {rows_examined} rows but returns "
                    f"{rows_returned}. Add more selective WHERE clause or index."
                )

            # Create analysis
            analysis = QueryAnalysis(
                query_hash=query_hash,
                query_text=query_text[:500],  # Truncate for storage
                execution_time_ms=execution_time_ms,
                plan_cost=plan_cost,
                rows_examined=rows_examined,
                rows_returned=rows_returned,
                uses_index=uses_index,
                full_table_scan=full_table_scan,
                recommendations=recommendations,
                timestamp=DeterministicClock.utcnow()
            )

            # Store in history
            self.query_history.append(analysis)

            # Log if slow or problematic
            if recommendations:
                logger.warning(
                    f"Query performance issue detected: {'; '.join(recommendations)}"
                )

            return analysis

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            raise

    def get_slow_queries(self, threshold_ms: Optional[float] = None) -> List[QueryAnalysis]:
        """Get all queries exceeding threshold"""
        threshold = threshold_ms or self.slow_query_threshold_ms
        return [
            q for q in self.query_history
            if q.execution_time_ms > threshold
        ]

    def get_full_table_scans(self) -> List[QueryAnalysis]:
        """Get all queries using full table scans"""
        return [q for q in self.query_history if q.full_table_scan]


# ============================================================================
# INDEX MANAGER
# ============================================================================

class IndexManager:
    """
    Manages database indexes for optimal query performance.

    Features:
    - Index creation from definitions
    - Index health monitoring
    - Missing index detection
    - Index usage statistics
    """

    def __init__(self, engine: AsyncEngine):
        """
        Initialize index manager.

        Args:
            engine: SQLAlchemy async engine
        """
        self.engine = engine

    async def create_indexes(
        self,
        indexes: Optional[List[Dict[str, Any]]] = None,
        force: bool = False
    ) -> Dict[str, bool]:
        """
        Create indexes from definitions.

        Args:
            indexes: List of index definitions (default: ALL_INDEXES)
            force: Whether to drop existing indexes first

        Returns:
            Dictionary of index_name -> success status
        """
        indexes = indexes or ALL_INDEXES
        results = {}

        async with self.engine.begin() as conn:
            for idx_def in indexes:
                try:
                    index_name = idx_def["name"]
                    table_name = idx_def["table"]
                    columns = idx_def["columns"]

                    # Check if index exists
                    check_query = text("""
                        SELECT indexname FROM pg_indexes
                        WHERE indexname = :index_name
                    """)
                    result = await conn.execute(
                        check_query,
                        {"index_name": index_name}
                    )
                    exists = result.fetchone() is not None

                    if exists and not force:
                        logger.info(f"Index {index_name} already exists, skipping")
                        results[index_name] = True
                        continue

                    if exists and force:
                        drop_query = f"DROP INDEX IF EXISTS {index_name}"
                        await conn.execute(text(drop_query))
                        logger.info(f"Dropped existing index {index_name}")

                    # Build CREATE INDEX statement
                    columns_str = ", ".join(columns)

                    # Check for covering index (INCLUDE clause)
                    include_clause = ""
                    if "include" in idx_def:
                        include_cols = ", ".join(idx_def["include"])
                        include_clause = f" INCLUDE ({include_cols})"

                    # Check for partial index (WHERE clause)
                    where_clause = ""
                    if "where" in idx_def:
                        where_clause = f" WHERE {idx_def['where']}"

                    create_query = f"""
                        CREATE INDEX {index_name}
                        ON {table_name} ({columns_str})
                        {include_clause}
                        {where_clause}
                    """

                    await conn.execute(text(create_query))
                    logger.info(
                        f"Created index {index_name} on {table_name}({columns_str})"
                    )
                    results[index_name] = True

                except Exception as e:
                    logger.error(f"Failed to create index {idx_def['name']}: {e}")
                    results[idx_def["name"]] = False

        return results

    async def get_index_stats(self) -> List[Dict[str, Any]]:
        """
        Get index usage statistics from PostgreSQL.

        Returns:
            List of index statistics including scans, size, and cache hit ratio
        """
        query = text("""
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan as scans,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched,
                pg_size_pretty(pg_relation_size(indexrelid)) as size,
                CASE
                    WHEN idx_scan = 0 THEN 'NEVER_USED'
                    WHEN idx_scan < 100 THEN 'RARELY_USED'
                    WHEN idx_scan < 1000 THEN 'MODERATELY_USED'
                    ELSE 'FREQUENTLY_USED'
                END as usage_level
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            ORDER BY idx_scan DESC
        """)

        async with self.engine.connect() as conn:
            result = await conn.execute(query)
            rows = result.fetchall()

            return [
                {
                    "schema": row[0],
                    "table": row[1],
                    "index": row[2],
                    "scans": row[3],
                    "tuples_read": row[4],
                    "tuples_fetched": row[5],
                    "size": row[6],
                    "usage_level": row[7]
                }
                for row in rows
            ]

    async def find_missing_indexes(self) -> List[Dict[str, Any]]:
        """
        Find potentially missing indexes based on query patterns.

        Uses PostgreSQL's pg_stat_statements to identify frequently
        scanned tables without appropriate indexes.

        Returns:
            List of recommendations for missing indexes
        """
        query = text("""
            SELECT
                schemaname,
                tablename,
                seq_scan,
                seq_tup_read,
                idx_scan,
                CASE
                    WHEN seq_scan > 0 AND idx_scan = 0 THEN 'CRITICAL'
                    WHEN seq_scan > idx_scan * 10 THEN 'HIGH'
                    WHEN seq_scan > idx_scan THEN 'MEDIUM'
                    ELSE 'LOW'
                END as priority
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
                AND seq_scan > 0
            ORDER BY seq_scan DESC
            LIMIT 20
        """)

        async with self.engine.connect() as conn:
            result = await conn.execute(query)
            rows = result.fetchall()

            recommendations = []
            for row in rows:
                if row[5] in ('CRITICAL', 'HIGH'):
                    recommendations.append({
                        "table": row[1],
                        "sequential_scans": row[2],
                        "tuples_scanned": row[3],
                        "index_scans": row[4],
                        "priority": row[5],
                        "recommendation": (
                            f"Table '{row[1]}' has {row[2]} sequential scans "
                            f"reading {row[3]} tuples. Consider adding indexes "
                            f"for frequently filtered columns."
                        )
                    })

            return recommendations


# ============================================================================
# N+1 QUERY OPTIMIZER
# ============================================================================

class NPlusOneOptimizer:
    """
    Detects and fixes N+1 query problems.

    N+1 Problem: Executing N additional queries for each result in initial query
    Solution: Use JOIN or eager loading (joinedload/selectinload)
    """

    @staticmethod
    def optimize_with_joinedload(query: Select, *relationships) -> Select:
        """
        Optimize query using joinedload (single JOIN query).

        Use for:
        - One-to-one relationships
        - Small to medium result sets
        - When you need all related data

        Args:
            query: Base query
            relationships: Relationship attributes to eager load

        Returns:
            Optimized query with joinedload
        """
        for rel in relationships:
            query = query.options(joinedload(rel))
        return query

    @staticmethod
    def optimize_with_selectinload(query: Select, *relationships) -> Select:
        """
        Optimize query using selectinload (separate SELECT IN query).

        Use for:
        - One-to-many relationships
        - Large result sets
        - When JOIN would create too many rows

        Args:
            query: Base query
            relationships: Relationship attributes to eager load

        Returns:
            Optimized query with selectinload
        """
        for rel in relationships:
            query = query.options(selectinload(rel))
        return query


# ============================================================================
# QUERY OPTIMIZATION UTILITIES
# ============================================================================

@asynccontextmanager
async def track_query_performance(
    query_name: str,
    log_threshold_ms: float = 100.0
):
    """
    Context manager to track query execution time.

    Usage:
        async with track_query_performance("get_emissions", log_threshold_ms=50):
            result = await session.execute(query)
    """
    start_time = time.time()

    try:
        yield
    finally:
        execution_time_ms = (time.time() - start_time) * 1000

        if execution_time_ms > log_threshold_ms:
            logger.warning(
                f"Slow query detected: {query_name} took {execution_time_ms:.2f}ms "
                f"(threshold: {log_threshold_ms}ms)"
            )
        else:
            logger.debug(
                f"Query {query_name} completed in {execution_time_ms:.2f}ms"
            )


async def batch_insert_optimized(
    session: AsyncSession,
    table_name: str,
    records: List[Dict[str, Any]],
    batch_size: int = 1000
) -> int:
    """
    Optimized batch insert using bulk_insert_mappings.

    Performance: 10-100x faster than individual inserts

    Args:
        session: Database session
        table_name: Target table name
        records: List of record dictionaries
        batch_size: Number of records per batch

    Returns:
        Number of records inserted
    """
    from sqlalchemy.orm import class_mapper

    total_inserted = 0

    # Process in batches
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]

        # Use bulk_insert_mappings for optimal performance
        await session.execute(
            sa.insert(table_name),
            batch
        )

        total_inserted += len(batch)

        if total_inserted % 10000 == 0:
            logger.info(f"Inserted {total_inserted}/{len(records)} records")

    await session.commit()
    logger.info(f"Batch insert completed: {total_inserted} records")

    return total_inserted


# ============================================================================
# SQL OPTIMIZATION TEMPLATES
# ============================================================================

SQL_OPTIMIZATION_EXAMPLES = """
-- ============================================================================
-- OPTIMIZED QUERY EXAMPLES
-- ============================================================================

-- Example 1: Composite Index Usage
-- BEFORE (slow - multiple index lookups)
SELECT * FROM emissions
WHERE supplier_id = '123'
  AND scope3_category = 'CATEGORY_1'
  AND transaction_date >= '2024-01-01';

-- AFTER (fast - single composite index lookup using idx_emissions_composite)
-- Query automatically uses idx_emissions_composite(supplier_id, scope3_category, transaction_date)


-- Example 2: Covering Index (Index-Only Scan)
-- BEFORE (slow - table access required)
SELECT transaction_date, scope3_category, emissions_tco2e, dqi_score
FROM emissions
WHERE transaction_date >= '2024-01-01'
  AND scope3_category = 'CATEGORY_1';

-- AFTER (fast - uses idx_emissions_reporting covering index, no table access)
-- Index includes all required columns: (transaction_date, scope3_category) INCLUDE (emissions_tco2e, dqi_score)


-- Example 3: Partial Index for Sparse Data
-- BEFORE (slow - index includes NULL values)
CREATE INDEX idx_suppliers_duns_old ON suppliers(duns_number);

-- AFTER (fast - partial index only for non-NULL values, smaller index)
CREATE INDEX idx_suppliers_duns ON suppliers(duns_number)
WHERE duns_number IS NOT NULL;


-- Example 4: Efficient JOIN vs N+1
-- BEFORE (N+1 problem - 1 query + N supplier queries)
-- SELECT * FROM emissions;  -- Returns N rows
-- for each emission:
--     SELECT * FROM suppliers WHERE id = emission.supplier_id;  -- N queries

-- AFTER (single JOIN query)
SELECT e.*, s.*
FROM emissions e
JOIN suppliers s ON e.supplier_id = s.id
WHERE e.transaction_date >= '2024-01-01';


-- Example 5: Aggregation with Index
-- BEFORE (slow - full table scan)
SELECT
    scope3_category,
    SUM(emissions_tco2e) as total_emissions
FROM emissions
GROUP BY scope3_category;

-- AFTER (fast - uses idx_emissions_category)
-- Query automatically uses index for grouping


-- Example 6: Multi-tenant Query Optimization
-- BEFORE (slow - filters after retrieving all data)
SELECT * FROM emissions
WHERE tenant_id = 'tenant-123'
  AND transaction_date >= '2024-01-01'
ORDER BY transaction_date DESC
LIMIT 100;

-- AFTER (fast - uses idx_emissions_tenant composite index)
-- Index: (tenant_id, transaction_date)


-- ============================================================================
-- QUERY ANALYSIS COMMANDS
-- ============================================================================

-- Analyze query plan (without execution)
EXPLAIN
SELECT * FROM emissions
WHERE supplier_id = '123'
  AND transaction_date >= '2024-01-01';

-- Analyze query with actual execution
EXPLAIN ANALYZE
SELECT * FROM emissions
WHERE supplier_id = '123'
  AND transaction_date >= '2024-01-01';

-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- Find missing indexes (tables with many sequential scans)
SELECT
    schemaname,
    tablename,
    seq_scan as sequential_scans,
    seq_tup_read as tuples_read,
    idx_scan as index_scans,
    CASE
        WHEN seq_scan > idx_scan * 10 THEN 'CRITICAL'
        WHEN seq_scan > idx_scan THEN 'REVIEW'
        ELSE 'OK'
    END as recommendation
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY seq_scan DESC;

-- Table statistics
SELECT
    tablename,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE schemaname = 'public';
"""


__all__ = [
    "QueryAnalyzer",
    "QueryAnalysis",
    "IndexManager",
    "NPlusOneOptimizer",
    "track_query_performance",
    "batch_insert_optimized",
    "EMISSIONS_INDEXES",
    "SUPPLIERS_INDEXES",
    "CALCULATIONS_INDEXES",
    "ALL_INDEXES",
]
