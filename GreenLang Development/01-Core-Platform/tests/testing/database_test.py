# -*- coding: utf-8 -*-
"""
Database Testing Framework
=========================

Test cases and utilities for testing database operations.

This module provides specialized test cases for testing database operations,
transactions, migrations, and query performance.
"""

import unittest
from typing import Any, Dict, List, Optional, Type
from unittest.mock import Mock, patch
from contextlib import contextmanager
import time

from .mocks import MockDatabaseManager


class DatabaseTestCase(unittest.TestCase):
    """
    Base test case for testing database operations.

    Provides test database setup/teardown, transaction rollback,
    mock database for unit tests, and query performance testing.

    Example:
    --------
    ```python
    class TestDatabase(DatabaseTestCase):
        def test_insert_and_query(self):
            with self.db_transaction():
                self.db.insert("users", {"name": "John", "email": "john@example.com"})
                result = self.db.query("SELECT * FROM users WHERE name = 'John'")
                self.assertEqual(len(result), 1)
                # Transaction automatically rolled back after test
    ```
    """

    def setUp(self):
        """Set up test database and fixtures."""
        self.db = MockDatabaseManager()

        # Track database operations
        self.db_operations = []
        self.query_times = []

        # Set up test schema
        self.setup_test_schema()

    def tearDown(self):
        """Clean up test database."""
        self.db.reset()
        self.db_operations.clear()

    def setup_test_schema(self):
        """Set up test database schema."""
        # This is a hook for subclasses to define their schema
        pass

    @contextmanager
    def db_transaction(self):
        """
        Context manager for database transactions with automatic rollback.

        All changes made within this context are rolled back after the test.

        Example:
        --------
        ```python
        with self.db_transaction():
            self.db.insert("users", {"name": "John"})
            # Changes rolled back automatically
        ```
        """
        # Start transaction
        self.db.begin_transaction()

        try:
            yield
        finally:
            # Always rollback
            self.db.rollback()

    @contextmanager
    def db_commit_transaction(self):
        """
        Context manager for database transactions with commit.

        Use this when you actually want to commit changes.

        Example:
        --------
        ```python
        with self.db_commit_transaction():
            self.db.insert("users", {"name": "John"})
            # Changes committed
        ```
        """
        # Start transaction
        self.db.begin_transaction()

        try:
            yield
            # Commit if no errors
            self.db.commit()
        except Exception:
            # Rollback on error
            self.db.rollback()
            raise

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Execute a query and track performance.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Tuple of (results, execution_time)
        """
        start = time.time()
        results = self.db.query(query, params or {})
        exec_time = time.time() - start

        self.db_operations.append({
            'type': 'query',
            'query': query,
            'params': params,
            'num_results': len(results) if results else 0,
            'time': exec_time,
        })

        self.query_times.append(exec_time)
        return results, exec_time

    def execute_insert(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> tuple[Any, float]:
        """
        Execute an insert and track performance.

        Args:
            table: Table name
            data: Data to insert

        Returns:
            Tuple of (inserted_id, execution_time)
        """
        start = time.time()
        result = self.db.insert(table, data)
        exec_time = time.time() - start

        self.db_operations.append({
            'type': 'insert',
            'table': table,
            'data': data,
            'time': exec_time,
        })

        return result, exec_time

    def execute_update(
        self,
        table: str,
        data: Dict[str, Any],
        where: Dict[str, Any]
    ) -> tuple[int, float]:
        """
        Execute an update and track performance.

        Args:
            table: Table name
            data: Data to update
            where: WHERE clause conditions

        Returns:
            Tuple of (num_updated, execution_time)
        """
        start = time.time()
        result = self.db.update(table, data, where)
        exec_time = time.time() - start

        self.db_operations.append({
            'type': 'update',
            'table': table,
            'data': data,
            'where': where,
            'time': exec_time,
        })

        return result, exec_time

    def execute_delete(
        self,
        table: str,
        where: Dict[str, Any]
    ) -> tuple[int, float]:
        """
        Execute a delete and track performance.

        Args:
            table: Table name
            where: WHERE clause conditions

        Returns:
            Tuple of (num_deleted, execution_time)
        """
        start = time.time()
        result = self.db.delete(table, where)
        exec_time = time.time() - start

        self.db_operations.append({
            'type': 'delete',
            'table': table,
            'where': where,
            'time': exec_time,
        })

        return result, exec_time

    def assert_query_result_count(
        self,
        query: str,
        expected_count: int,
        params: Optional[Dict[str, Any]] = None
    ):
        """Assert that a query returns expected number of results."""
        results, _ = self.execute_query(query, params)
        self.assertEqual(
            len(results),
            expected_count,
            f"Query returned {len(results)} results, expected {expected_count}"
        )

    def assert_query_performance(
        self,
        query: str,
        max_time: float,
        params: Optional[Dict[str, Any]] = None
    ):
        """Assert that a query executes within time limit."""
        _, exec_time = self.execute_query(query, params)
        self.assertLessEqual(
            exec_time,
            max_time,
            f"Query took {exec_time:.4f}s, exceeded max {max_time}s"
        )

    def assert_record_exists(
        self,
        table: str,
        where: Dict[str, Any]
    ):
        """Assert that a record exists in the database."""
        where_clause = " AND ".join([f"{k} = :{k}" for k in where.keys()])
        query = f"SELECT * FROM {table} WHERE {where_clause}"
        results, _ = self.execute_query(query, where)

        self.assertTrue(
            len(results) > 0,
            f"No record found in {table} with {where}"
        )

    def assert_record_not_exists(
        self,
        table: str,
        where: Dict[str, Any]
    ):
        """Assert that a record does not exist in the database."""
        where_clause = " AND ".join([f"{k} = :{k}" for k in where.keys()])
        query = f"SELECT * FROM {table} WHERE {where_clause}"
        results, _ = self.execute_query(query, where)

        self.assertEqual(
            len(results),
            0,
            f"Record unexpectedly found in {table} with {where}"
        )

    def assert_record_count(
        self,
        table: str,
        expected_count: int,
        where: Optional[Dict[str, Any]] = None
    ):
        """Assert the number of records in a table."""
        if where:
            where_clause = " WHERE " + " AND ".join([f"{k} = :{k}" for k in where.keys()])
            query = f"SELECT COUNT(*) as count FROM {table}{where_clause}"
            results, _ = self.execute_query(query, where)
        else:
            query = f"SELECT COUNT(*) as count FROM {table}"
            results, _ = self.execute_query(query)

        count = results[0]['count'] if results else 0
        self.assertEqual(
            count,
            expected_count,
            f"Table {table} has {count} records, expected {expected_count}"
        )

    def load_fixtures(self, fixtures: Dict[str, List[Dict[str, Any]]]):
        """
        Load test fixtures into database.

        Args:
            fixtures: Dict mapping table names to lists of records

        Example:
        --------
        ```python
        self.load_fixtures({
            'users': [
                {'name': 'John', 'email': 'john@example.com'},
                {'name': 'Jane', 'email': 'jane@example.com'},
            ],
            'products': [
                {'name': 'Product 1', 'price': 100},
            ]
        })
        ```
        """
        for table, records in fixtures.items():
            for record in records:
                self.db.insert(table, record)

    def get_db_stats(self) -> Dict[str, Any]:
        """Get aggregated database statistics."""
        return {
            'total_operations': len(self.db_operations),
            'queries': len([op for op in self.db_operations if op['type'] == 'query']),
            'inserts': len([op for op in self.db_operations if op['type'] == 'insert']),
            'updates': len([op for op in self.db_operations if op['type'] == 'update']),
            'deletes': len([op for op in self.db_operations if op['type'] == 'delete']),
            'avg_query_time': sum(self.query_times) / max(len(self.query_times), 1),
            'total_query_time': sum(self.query_times),
        }
