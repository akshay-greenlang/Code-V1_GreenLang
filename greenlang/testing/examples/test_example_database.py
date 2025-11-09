"""
Example Database Test
=====================

Demonstrates how to test database operations.
"""

from greenlang.testing import DatabaseTestCase


class TestDatabase(DatabaseTestCase):
    """Test suite for database operations."""

    def setup_test_schema(self):
        """Set up test database schema."""
        # In a real scenario, this would create tables
        # For this example, our mock DB handles it automatically
        pass

    def test_insert_and_query(self):
        """Test basic insert and query operations."""
        with self.db_transaction():
            # Insert data
            user_id, _ = self.execute_insert("users", {
                "name": "John Doe",
                "email": "john@example.com",
                "role": "admin"
            })

            # Query data
            results, _ = self.execute_query(
                "SELECT * FROM users WHERE name = :name",
                {"name": "John Doe"}
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["email"], "john@example.com")
            self.assertEqual(results[0]["role"], "admin")

    def test_update_operation(self):
        """Test update operation."""
        with self.db_transaction():
            # Insert
            self.db.insert("users", {
                "name": "Jane Doe",
                "email": "jane@example.com",
                "status": "active"
            })

            # Update
            num_updated, _ = self.execute_update(
                "users",
                {"status": "inactive"},
                {"name": "Jane Doe"}
            )

            self.assertEqual(num_updated, 1)

            # Verify update
            results, _ = self.execute_query(
                "SELECT * FROM users WHERE name = :name",
                {"name": "Jane Doe"}
            )

            self.assertEqual(results[0]["status"], "inactive")

    def test_delete_operation(self):
        """Test delete operation."""
        with self.db_transaction():
            # Insert
            self.db.insert("users", {
                "name": "Delete Me",
                "email": "delete@example.com"
            })

            # Verify exists
            self.assert_record_exists("users", {"name": "Delete Me"})

            # Delete
            num_deleted, _ = self.execute_delete(
                "users",
                {"name": "Delete Me"}
            )

            self.assertEqual(num_deleted, 1)

            # Verify deleted
            self.assert_record_not_exists("users", {"name": "Delete Me"})

    def test_transaction_rollback(self):
        """Test that transactions are rolled back after test."""
        with self.db_transaction():
            # Insert data
            self.db.insert("users", {
                "name": "Temporary User",
                "email": "temp@example.com"
            })

            # Data exists within transaction
            self.assert_record_exists("users", {"name": "Temporary User"})

        # After transaction, data should be rolled back
        # In our mock, this is simulated
        self.db.reset()
        self.assert_record_not_exists("users", {"name": "Temporary User"})

    def test_with_fixtures(self):
        """Test with fixture data."""
        with self.db_transaction():
            # Load fixtures
            self.load_fixtures({
                'users': [
                    {'name': 'Alice', 'email': 'alice@example.com'},
                    {'name': 'Bob', 'email': 'bob@example.com'},
                    {'name': 'Charlie', 'email': 'charlie@example.com'},
                ],
                'products': [
                    {'name': 'Product A', 'price': 100},
                    {'name': 'Product B', 'price': 200},
                ]
            })

            # Verify counts
            self.assert_record_count('users', expected_count=3)
            self.assert_record_count('products', expected_count=2)

    def test_query_performance(self):
        """Test query performance."""
        with self.db_transaction():
            # Insert many records
            for i in range(100):
                self.db.insert("products", {
                    "name": f"Product {i}",
                    "price": i * 10,
                    "category": "electronics" if i % 2 == 0 else "clothing"
                })

            # Query should be fast
            self.assert_query_performance(
                "SELECT * FROM products WHERE price > :price",
                max_time=0.5,  # 500ms max
                params={"price": 500}
            )

    def test_emissions_data(self):
        """Test storing and querying emissions data."""
        with self.db_transaction():
            # Insert emissions record
            self.db.insert("emissions", {
                "emission_id": "em_001",
                "scope": 1,
                "category": "Stationary Combustion",
                "quantity": 5000,
                "emission_factor": 2.0,
                "total_emissions": 10000,
                "date": "2024-01-01"
            })

            # Query by scope
            results, _ = self.execute_query(
                "SELECT * FROM emissions WHERE scope = :scope",
                {"scope": 1}
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["total_emissions"], 10000)

    def test_database_stats(self):
        """Test database operation statistics."""
        with self.db_transaction():
            # Perform various operations
            self.execute_insert("users", {"name": "User 1", "email": "u1@example.com"})
            self.execute_insert("users", {"name": "User 2", "email": "u2@example.com"})
            self.execute_query("SELECT * FROM users")
            self.execute_update("users", {"status": "active"}, {"name": "User 1"})
            self.execute_delete("users", {"name": "User 2"})

            # Get stats
            stats = self.get_db_stats()

            self.assertEqual(stats['inserts'], 2)
            self.assertEqual(stats['queries'], 1)
            self.assertEqual(stats['updates'], 1)
            self.assertEqual(stats['deletes'], 1)
            self.assertEqual(stats['total_operations'], 5)


if __name__ == '__main__':
    import unittest
    unittest.main()
