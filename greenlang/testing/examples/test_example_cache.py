"""
Example Cache Test
==================

Demonstrates how to test caching.
"""

from greenlang.testing import CacheTestCase
import time


class TestCaching(CacheTestCase):
    """Test suite for caching functionality."""

    def test_basic_cache_operations(self):
        """Test basic set and get operations."""
        # Set value
        self.set_cache("emissions_key", {"total": 2500, "unit": "kg CO2e"})

        # Get value
        value, exec_time, is_hit = self.get_cache("emissions_key")

        self.assertIsNotNone(value)
        self.assertEqual(value['total'], 2500)
        self.assertTrue(is_hit)

    def test_cache_miss(self):
        """Test cache miss scenario."""
        value, exec_time, is_miss = self.get_cache("nonexistent_key")

        self.assertIsNone(value)
        self.assertFalse(is_miss)  # is_hit is False, so it's a miss

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        # Set some values
        self.set_cache("key1", "value1")
        self.set_cache("key2", "value2")
        self.set_cache("key3", "value3")

        # Generate hits and misses
        self.get_cache("key1")  # Hit
        self.get_cache("key2")  # Hit
        self.get_cache("key3")  # Hit
        self.get_cache("key4")  # Miss
        self.get_cache("key5")  # Miss

        # 3 hits, 2 misses = 60% hit rate
        self.assert_hit_rate(min_rate=0.5, max_rate=0.7)

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        # Set with 1 second TTL
        self.set_cache("temp_key", "temp_value", ttl=1)

        # Should exist immediately
        self.assert_cache_hit("temp_key")

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        self.assert_cache_miss("temp_key")

    def test_cache_update(self):
        """Test updating cached values."""
        # Set initial value
        self.set_cache("update_key", "initial_value")

        # Verify
        value, _, _ = self.get_cache("update_key")
        self.assertEqual(value, "initial_value")

        # Update
        self.set_cache("update_key", "updated_value")

        # Verify update
        value, _, _ = self.get_cache("update_key")
        self.assertEqual(value, "updated_value")

    def test_cache_deletion(self):
        """Test cache deletion."""
        # Set value
        self.set_cache("delete_key", "delete_value")

        # Verify exists
        self.assert_cache_hit("delete_key")

        # Delete
        self.delete_cache("delete_key")

        # Verify deleted
        self.assert_cache_miss("delete_key")

    def test_cache_performance(self):
        """Test cache operation performance."""
        # Perform many operations
        for i in range(100):
            self.set_cache(f"perf_key_{i}", f"perf_value_{i}")
            self.get_cache(f"perf_key_{i}")

        # Average operation should be very fast
        self.assert_cache_performance(max_avg_time=0.01)  # 10ms max

    def test_cache_under_load(self):
        """Test cache under simulated load."""
        # Simulate 1000 operations with 70% hit probability
        self.simulate_cache_load(
            num_operations=1000,
            hit_probability=0.7
        )

        # Get stats
        stats = self.get_cache_stats()

        # Should have reasonable hit rate
        self.assertGreater(stats['hit_rate'], 0.6)

        # Should be performant
        self.assertLess(stats['avg_operation_time'], 0.01)

    def test_complex_data_caching(self):
        """Test caching complex data structures."""
        complex_data = {
            'emissions': [
                {'scope': 1, 'total': 10000},
                {'scope': 2, 'total': 4000},
                {'scope': 3, 'total': 2860},
            ],
            'metadata': {
                'period': 'Q1 2024',
                'verified': True,
            }
        }

        self.set_cache("complex_key", complex_data)

        value, _, _ = self.get_cache("complex_key")

        self.assertEqual(value['emissions'][0]['total'], 10000)
        self.assertEqual(value['metadata']['period'], 'Q1 2024')


if __name__ == '__main__':
    import unittest
    unittest.main()
