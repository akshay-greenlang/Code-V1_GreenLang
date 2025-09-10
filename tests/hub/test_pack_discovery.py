"""
Tests for Pack Discovery and Index System
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

from greenlang.hub.index import (
    PackIndex, PackInfo, SearchFilters, SortOrder, PackCategory
)


class TestPackInfo(unittest.TestCase):
    """Test PackInfo dataclass"""
    
    def test_pack_info_creation(self):
        """Test creating PackInfo object"""
        pack = PackInfo(
            id="test-pack-1",
            name="test-pack",
            version="1.0.0",
            description="Test pack description",
            author={"name": "Test Author", "email": "test@example.com"},
            categories=["development", "testing"],
            tags=["test", "example"],
            downloads=1000,
            stars=50,
            verified=True,
            official=False
        )
        
        self.assertEqual(pack.id, "test-pack-1")
        self.assertEqual(pack.name, "test-pack")
        self.assertEqual(pack.version, "1.0.0")
        self.assertEqual(pack.downloads, 1000)
        self.assertTrue(pack.verified)
        self.assertFalse(pack.official)
    
    def test_pack_info_to_dict(self):
        """Test converting PackInfo to dictionary"""
        pack = PackInfo(
            id="test-pack-1",
            name="test-pack",
            version="1.0.0",
            description="Test pack",
            author={"name": "Author"},
            created_at=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        data = pack.to_dict()
        
        self.assertEqual(data['id'], "test-pack-1")
        self.assertEqual(data['name'], "test-pack")
        self.assertIn('created_at', data)
        self.assertIsInstance(data['created_at'], str)
    
    def test_pack_info_from_dict(self):
        """Test creating PackInfo from dictionary"""
        data = {
            "id": "test-pack-1",
            "name": "test-pack",
            "version": "1.0.0",
            "description": "Test pack",
            "author": {"name": "Author"},
            "created_at": "2024-01-01T12:00:00",
            "downloads": 500
        }
        
        pack = PackInfo.from_dict(data)
        
        self.assertEqual(pack.id, "test-pack-1")
        self.assertEqual(pack.downloads, 500)
        self.assertIsInstance(pack.created_at, datetime)


class TestSearchFilters(unittest.TestCase):
    """Test SearchFilters functionality"""
    
    def test_filters_to_params(self):
        """Test converting filters to query parameters"""
        filters = SearchFilters(
            categories=["development", "testing"],
            tags=["python", "cli"],
            author="testuser",
            min_stars=10,
            verified_only=True,
            exclude_deprecated=True
        )
        
        params = filters.to_params()
        
        self.assertEqual(params['categories'], "development,testing")
        self.assertEqual(params['tags'], "python,cli")
        self.assertEqual(params['author'], "testuser")
        self.assertEqual(params['min_stars'], 10)
        self.assertEqual(params['verified'], 'true')
        self.assertEqual(params['exclude_deprecated'], 'true')
    
    def test_empty_filters(self):
        """Test empty filters"""
        filters = SearchFilters()
        params = filters.to_params()
        
        # Should only have exclude_deprecated by default
        self.assertEqual(params.get('exclude_deprecated'), 'true')
        self.assertNotIn('categories', params)
        self.assertNotIn('author', params)


class TestPackIndex(unittest.TestCase):
    """Test PackIndex functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Mock HubClient
        self.mock_client = Mock()
        self.mock_session = MagicMock()
        self.mock_client.session = self.mock_session
        
        # Create test packs
        self.test_packs = [
            {
                "id": "pack1",
                "name": "test-pack-1",
                "version": "1.0.0",
                "description": "First test pack",
                "author": {"name": "Author1"},
                "categories": ["development"],
                "tags": ["test", "example"],
                "downloads": 1000,
                "stars": 50,
                "verified": True
            },
            {
                "id": "pack2",
                "name": "test-pack-2",
                "version": "2.0.0",
                "description": "Second test pack",
                "author": {"name": "Author2"},
                "categories": ["testing"],
                "tags": ["test"],
                "downloads": 500,
                "stars": 25,
                "official": True
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_search(self):
        """Test search functionality"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": self.test_packs}
        mock_response.raise_for_status = MagicMock()
        self.mock_session.get.return_value = mock_response
        
        # Test search
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        results = index.search(query="test", limit=10)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].name, "test-pack-1")
        self.assertEqual(results[1].name, "test-pack-2")
        
        # Verify API call
        self.mock_session.get.assert_called_once()
        call_args = self.mock_session.get.call_args
        self.assertEqual(call_args[0][0], "/api/v1/search")
        self.assertIn("q", call_args[1]["params"])
    
    def test_search_with_filters(self):
        """Test search with filters"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [self.test_packs[0]]}
        mock_response.raise_for_status = MagicMock()
        self.mock_session.get.return_value = mock_response
        
        # Test search with filters
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        filters = SearchFilters(
            categories=["development"],
            min_stars=40,
            verified_only=True
        )
        results = index.search(filters=filters, sort=SortOrder.STARS)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "test-pack-1")
        
        # Verify parameters
        call_args = self.mock_session.get.call_args
        params = call_args[1]["params"]
        self.assertEqual(params["sort"], "stars")
        self.assertEqual(params["categories"], "development")
        self.assertEqual(params["min_stars"], 40)
    
    def test_get_featured(self):
        """Test getting featured packs"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = self.test_packs[:1]
        mock_response.raise_for_status = MagicMock()
        self.mock_session.get.return_value = mock_response
        
        # Test get featured
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        featured = index.get_featured(limit=5)
        
        self.assertEqual(len(featured), 1)
        self.assertEqual(featured[0].name, "test-pack-1")
        
        # Verify caching
        cached = index.get_featured(limit=5)
        self.assertEqual(len(cached), 1)
        # Should only be called once due to caching
        self.mock_session.get.assert_called_once()
    
    def test_get_trending(self):
        """Test getting trending packs"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = self.test_packs
        mock_response.raise_for_status = MagicMock()
        self.mock_session.get.return_value = mock_response
        
        # Test get trending
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        trending = index.get_trending(period="week", limit=10)
        
        self.assertEqual(len(trending), 2)
        
        # Verify API call
        call_args = self.mock_session.get.call_args
        self.assertEqual(call_args[0][0], "/api/v1/trending")
        self.assertEqual(call_args[1]["params"]["period"], "week")
    
    def test_get_by_category(self):
        """Test getting packs by category"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [self.test_packs[0]]}
        mock_response.raise_for_status = MagicMock()
        self.mock_session.get.return_value = mock_response
        
        # Test get by category
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        packs = index.get_by_category(PackCategory.DEVELOPMENT, limit=10)
        
        self.assertEqual(len(packs), 1)
        self.assertEqual(packs[0].name, "test-pack-1")
        
        # Verify filter was applied
        call_args = self.mock_session.get.call_args
        params = call_args[1]["params"]
        self.assertEqual(params["categories"], "development")
    
    def test_get_similar(self):
        """Test getting similar packs"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = [self.test_packs[1]]
        mock_response.raise_for_status = MagicMock()
        self.mock_session.get.return_value = mock_response
        
        # Test get similar
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        similar = index.get_similar("pack1", limit=5)
        
        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0].name, "test-pack-2")
        
        # Verify API call
        call_args = self.mock_session.get.call_args
        self.assertEqual(call_args[0][0], "/api/v1/packs/pack1/similar")
    
    def test_get_recommendations(self):
        """Test getting personalized recommendations"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = self.test_packs
        mock_response.raise_for_status = MagicMock()
        self.mock_session.post.return_value = mock_response
        
        # Test recommendations
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        recommendations = index.get_recommendations(
            user_packs=["existing-pack-1"],
            limit=10
        )
        
        self.assertEqual(len(recommendations), 2)
        
        # Verify API call
        call_args = self.mock_session.post.call_args
        self.assertEqual(call_args[0][0], "/api/v1/recommendations")
        data = call_args[1]["json"]
        self.assertIn("existing-pack-1", data["installed_packs"])
    
    def test_local_index_caching(self):
        """Test local index caching and offline search"""
        # Create local index file
        index_file = self.temp_path / "index.json"
        index_data = {"packs": self.test_packs}
        with open(index_file, 'w') as f:
            json.dump(index_data, f)
        
        # Test loading local index
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        
        # Test local search
        results = index.search_local(query="test")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].name, "test-pack-1")
        self.assertEqual(results[1].name, "test-pack-2")
    
    def test_local_search_with_filters(self):
        """Test local search with filters"""
        # Create local index
        index_file = self.temp_path / "index.json"
        index_data = {"packs": self.test_packs}
        with open(index_file, 'w') as f:
            json.dump(index_data, f)
        
        # Test local search with filters
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        filters = SearchFilters(
            categories=["development"],
            verified_only=True
        )
        results = index.search_local(filters=filters)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "test-pack-1")
        self.assertTrue(results[0].verified)
    
    def test_get_categories(self):
        """Test getting categories"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"name": "development", "display_name": "Development", "count": 100},
            {"name": "testing", "display_name": "Testing", "count": 50}
        ]
        mock_response.raise_for_status = MagicMock()
        self.mock_session.get.return_value = mock_response
        
        # Test get categories
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        categories = index.get_categories()
        
        self.assertEqual(len(categories), 2)
        self.assertEqual(categories[0]["name"], "development")
        self.assertEqual(categories[0]["count"], 100)
    
    def test_get_tags(self):
        """Test getting popular tags"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"name": "python", "count": 500},
            {"name": "cli", "count": 300},
            {"name": "test", "count": 200}
        ]
        mock_response.raise_for_status = MagicMock()
        self.mock_session.get.return_value = mock_response
        
        # Test get tags
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        tags = index.get_tags(limit=50)
        
        self.assertEqual(len(tags), 3)
        self.assertEqual(tags[0]["name"], "python")
        self.assertEqual(tags[0]["count"], 500)
    
    def test_cache_expiry(self):
        """Test cache expiry mechanism"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = self.test_packs[:1]
        mock_response.raise_for_status = MagicMock()
        self.mock_session.get.return_value = mock_response
        
        # Test with short TTL
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path, cache_ttl=0)
        
        # First call
        featured1 = index.get_featured(limit=5)
        self.assertEqual(len(featured1), 1)
        
        # Second call (cache expired)
        featured2 = index.get_featured(limit=5)
        self.assertEqual(len(featured2), 1)
        
        # Should be called twice due to expired cache
        self.assertEqual(self.mock_session.get.call_count, 2)
    
    def test_error_fallback_to_cache(self):
        """Test fallback to cache on API error"""
        # First successful call to populate cache
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": self.test_packs}
        mock_response.raise_for_status = MagicMock()
        self.mock_session.get.return_value = mock_response
        
        index = PackIndex(client=self.mock_client, cache_dir=self.temp_path)
        results1 = index.search(query="test")
        self.assertEqual(len(results1), 2)
        
        # Now simulate API error
        self.mock_session.get.side_effect = Exception("API Error")
        
        # Should return cached results
        results2 = index.search(query="test")
        self.assertEqual(len(results2), 2)


if __name__ == '__main__':
    unittest.main()