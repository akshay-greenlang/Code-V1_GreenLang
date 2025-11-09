"""
Database Query Profiler
=======================

Identify slow queries, missing indexes, and database bottlenecks.

Features:
- Query execution time tracking
- Slow query identification
- Missing index detection
- Connection pool monitoring
- Query plan analysis
- N+1 query detection

Usage:
    from profile_db import DBProfiler

    profiler = DBProfiler()

    # Profile queries
    with profiler.profile("get_shipments"):
        results = db.execute(query)

    # Generate report
    profiler.generate_report()

Author: Performance Engineering Team
Date: 2025-11-09
Version: 1.0.0
"""

import time
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict


@dataclass
class QueryExecution:
    """Record of a single query execution."""
    timestamp: float
    operation: str
    query: str
    params: Optional[Dict]
    duration_ms: float
    rows_affected: int
    connection_time_ms: float


class DBProfiler:
    """Profile database queries and connections."""

    def __init__(self, output_dir: Path = None, slow_query_threshold_ms: float = 100.0):
        """
        Initialize database profiler.

        Args:
            output_dir: Directory for output files
            slow_query_threshold_ms: Threshold for slow queries in milliseconds
        """
        self.output_dir = output_dir or Path("./profiles")
        self.output_dir.mkdir(exist_ok=True)

        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.executions: List[QueryExecution] = []
        self.active_operation: Optional[str] = None
        self.operation_start: Optional[float] = None

    @contextmanager
    def profile(self, operation: str):
        """
        Context manager to profile a database operation.

        Args:
            operation: Operation name

        Example:
            >>> with profiler.profile("get_shipments"):
            ...     results = db.execute(query)
        """
        self.active_operation = operation
        self.operation_start = time.perf_counter()

        try:
            yield self
        finally:
            self.active_operation = None
            self.operation_start = None

    def record_query(
        self,
        query: str,
        params: Optional[Dict] = None,
        rows_affected: int = 0,
        connection_time_ms: float = 0.0
    ):
        """
        Record a query execution.

        Args:
            query: SQL query string
            params: Query parameters
            rows_affected: Number of rows affected
            connection_time_ms: Time to acquire connection
        """
        # Calculate duration
        duration_ms = 0
        if self.operation_start:
            duration_ms = (time.perf_counter() - self.operation_start) * 1000

        execution = QueryExecution(
            timestamp=time.time(),
            operation=self.active_operation or "unknown",
            query=self._normalize_query(query),
            params=params,
            duration_ms=duration_ms,
            rows_affected=rows_affected,
            connection_time_ms=connection_time_ms
        )

        self.executions.append(execution)

        # Warn about slow queries
        if duration_ms > self.slow_query_threshold_ms:
            print(f"⚠️  Slow query detected ({duration_ms:.2f}ms): {query[:100]}...")

    def _normalize_query(self, query: str) -> str:
        """Normalize query for analysis (remove whitespace, etc)."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        return query

    def get_slow_queries(self, limit: int = 20) -> List[Dict]:
        """
        Get slow queries.

        Args:
            limit: Number of slow queries to return

        Returns:
            List of slow queries
        """
        slow_queries = [
            ex for ex in self.executions
            if ex.duration_ms > self.slow_query_threshold_ms
        ]

        # Sort by duration
        slow_queries.sort(key=lambda x: x.duration_ms, reverse=True)

        return [
            {
                "query": ex.query,
                "duration_ms": ex.duration_ms,
                "operation": ex.operation,
                "rows": ex.rows_affected
            }
            for ex in slow_queries[:limit]
        ]

    def get_query_patterns(self) -> Dict[str, Dict]:
        """
        Analyze query patterns.

        Returns:
            Dictionary of query patterns with stats
        """
        patterns = defaultdict(lambda: {
            "count": 0,
            "total_time_ms": 0,
            "avg_time_ms": 0,
            "max_time_ms": 0,
            "min_time_ms": float('inf')
        })

        for ex in self.executions:
            # Extract query type (SELECT, INSERT, UPDATE, DELETE)
            query_type = ex.query.split()[0].upper() if ex.query else "UNKNOWN"

            # Group by query type and table
            table = self._extract_table(ex.query)
            pattern_key = f"{query_type} {table}"

            stats = patterns[pattern_key]
            stats["count"] += 1
            stats["total_time_ms"] += ex.duration_ms
            stats["max_time_ms"] = max(stats["max_time_ms"], ex.duration_ms)
            stats["min_time_ms"] = min(stats["min_time_ms"], ex.duration_ms)

        # Calculate averages
        for pattern_key in patterns:
            stats = patterns[pattern_key]
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["count"]

        return dict(patterns)

    def _extract_table(self, query: str) -> str:
        """Extract main table name from query."""
        # Simple extraction (can be improved)
        match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
        if match:
            return match.group(1)

        match = re.search(r'INTO\s+(\w+)', query, re.IGNORECASE)
        if match:
            return match.group(1)

        match = re.search(r'UPDATE\s+(\w+)', query, re.IGNORECASE)
        if match:
            return match.group(1)

        return "unknown"

    def detect_n_plus_1(self) -> List[Dict]:
        """
        Detect N+1 query problems.

        Returns:
            List of potential N+1 issues
        """
        issues = []
        query_groups = defaultdict(list)

        # Group queries by normalized pattern
        for ex in self.executions:
            # Remove parameter values for grouping
            normalized = re.sub(r'=\s*[\'"]?\w+[\'"]?', '= ?', ex.query)
            query_groups[normalized].append(ex)

        # Check for repeated similar queries
        for pattern, executions in query_groups.items():
            if len(executions) > 10:  # More than 10 similar queries
                total_time = sum(ex.duration_ms for ex in executions)
                if total_time > 100:  # And taking significant time
                    issues.append({
                        "pattern": pattern,
                        "count": len(executions),
                        "total_time_ms": total_time,
                        "avg_time_ms": total_time / len(executions),
                        "recommendation": "Consider using JOIN or batch loading"
                    })

        return issues

    def get_connection_stats(self) -> Dict:
        """Get connection pool statistics."""
        if not self.executions:
            return {}

        connection_times = [ex.connection_time_ms for ex in self.executions if ex.connection_time_ms > 0]

        if not connection_times:
            return {}

        return {
            "avg_connection_time_ms": sum(connection_times) / len(connection_times),
            "max_connection_time_ms": max(connection_times),
            "min_connection_time_ms": min(connection_times),
            "total_connections": len(connection_times)
        }

    def get_missing_indexes(self) -> List[Dict]:
        """
        Suggest missing indexes based on query patterns.

        Returns:
            List of index suggestions
        """
        suggestions = []

        for ex in self.executions:
            if ex.duration_ms < self.slow_query_threshold_ms:
                continue

            # Check for WHERE clauses without indexes (simple heuristic)
            where_match = re.search(r'WHERE\s+(\w+)\s*=', ex.query, re.IGNORECASE)
            if where_match:
                column = where_match.group(1)
                table = self._extract_table(ex.query)

                suggestions.append({
                    "table": table,
                    "column": column,
                    "query": ex.query,
                    "duration_ms": ex.duration_ms,
                    "suggestion": f"CREATE INDEX idx_{table}_{column} ON {table}({column})"
                })

        # Deduplicate
        unique_suggestions = {}
        for suggestion in suggestions:
            key = (suggestion["table"], suggestion["column"])
            if key not in unique_suggestions or suggestion["duration_ms"] > unique_suggestions[key]["duration_ms"]:
                unique_suggestions[key] = suggestion

        return list(unique_suggestions.values())

    def generate_html_report(self, output_file: str = "db_profile_report.html"):
        """Generate comprehensive HTML database report."""
        slow_queries = self.get_slow_queries()
        query_patterns = self.get_query_patterns()
        n_plus_1_issues = self.detect_n_plus_1()
        missing_indexes = self.get_missing_indexes()
        connection_stats = self.get_connection_stats()

        total_queries = len(self.executions)
        total_time = sum(ex.duration_ms for ex in self.executions)
        avg_time = total_time / total_queries if total_queries > 0 else 0

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Database Profile Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #16a085;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .summary-item {{
            display: inline-block;
            margin-right: 30px;
        }}
        .summary-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .summary-value {{
            font-size: 1.5em;
            color: #2c3e50;
        }}
        .danger {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 10px 0;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: #16a085;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            display: block;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Database Profile Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary">
            <div class="summary-item">
                <div class="summary-label">Total Queries</div>
                <div class="summary-value">{total_queries}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Avg Query Time</div>
                <div class="summary-value">{avg_time:.2f}ms</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Slow Queries</div>
                <div class="summary-value">{len(slow_queries)}</div>
            </div>
        </div>
        """

        # Slow queries
        if slow_queries:
            html += f"""
        <h2>Slow Queries (> {self.slow_query_threshold_ms}ms)</h2>
        <table>
            <thead>
                <tr>
                    <th>Query</th>
                    <th>Duration (ms)</th>
                    <th>Operation</th>
                    <th>Rows</th>
                </tr>
            </thead>
            <tbody>
            """

            for query in slow_queries[:20]:
                html += f"""
                <tr>
                    <td><code>{query['query']}</code></td>
                    <td>{query['duration_ms']:.2f}</td>
                    <td>{query['operation']}</td>
                    <td>{query['rows']}</td>
                </tr>
                """

            html += """
            </tbody>
        </table>
        """

        # N+1 issues
        if n_plus_1_issues:
            html += "<h2>Potential N+1 Query Problems</h2>"

            for issue in n_plus_1_issues:
                html += f"""
                <div class="danger">
                    <strong>Repeated Query Pattern:</strong><br>
                    <code>{issue['pattern']}</code><br>
                    <strong>Count:</strong> {issue['count']} executions<br>
                    <strong>Total Time:</strong> {issue['total_time_ms']:.2f}ms<br>
                    <strong>Recommendation:</strong> {issue['recommendation']}
                </div>
                """

        # Missing indexes
        if missing_indexes:
            html += "<h2>Suggested Indexes</h2>"
            html += "<table><thead><tr><th>Table</th><th>Column</th><th>Query Time (ms)</th><th>Suggestion</th></tr></thead><tbody>"

            for suggestion in missing_indexes[:10]:
                html += f"""
                <tr>
                    <td>{suggestion['table']}</td>
                    <td>{suggestion['column']}</td>
                    <td>{suggestion['duration_ms']:.2f}</td>
                    <td><code>{suggestion['suggestion']}</code></td>
                </tr>
                """

            html += "</tbody></table>"

        html += """
    </div>
</body>
</html>
        """

        output_path = self.output_dir / output_file
        with open(output_path, "w") as f:
            f.write(html)

        print(f"Database profile report saved to: {output_path}")

    def print_summary(self):
        """Print database profile summary."""
        print("\n" + "=" * 80)
        print("DATABASE PROFILE SUMMARY")
        print("=" * 80)

        print(f"\nTotal Queries: {len(self.executions)}")

        if self.executions:
            total_time = sum(ex.duration_ms for ex in self.executions)
            avg_time = total_time / len(self.executions)
            print(f"Average Query Time: {avg_time:.2f}ms")

            slow_queries = self.get_slow_queries()
            print(f"Slow Queries: {len(slow_queries)}")

            n_plus_1 = self.detect_n_plus_1()
            if n_plus_1:
                print(f"\n⚠️  Detected {len(n_plus_1)} potential N+1 query issues")

            missing_indexes = self.get_missing_indexes()
            if missing_indexes:
                print(f"⚠️  Suggested {len(missing_indexes)} missing indexes")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Demo usage
    profiler = DBProfiler(slow_query_threshold_ms=50.0)

    # Simulate queries
    with profiler.profile("get_shipments"):
        profiler.record_query(
            "SELECT * FROM shipments WHERE country = 'US'",
            rows_affected=100
        )

    # Simulate slow query
    with profiler.profile("complex_aggregation"):
        profiler.record_query(
            "SELECT country, SUM(emissions) FROM shipments GROUP BY country",
            rows_affected=50
        )

    profiler.print_summary()
    profiler.generate_html_report()
