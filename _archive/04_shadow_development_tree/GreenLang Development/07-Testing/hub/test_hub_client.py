# -*- coding: utf-8 -*-
"""
Tests for GreenLang Hub Client
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import json
from pathlib import Path
import httpx

from greenlang.hub.client import HubClient
from greenlang.hub.auth import HubAuth, PackSigner
from greenlang.hub.manifest import PackManifest, create_manifest, load_manifest
from greenlang.hub.archive import create_pack_archive, extract_pack_archive


class TestHubClient(unittest.TestCase):
    """Test HubClient functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test manifest
        self.test_manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "description": "Test pack for unit tests",
            "author": {"name": "Test Author"},
            "modules": ["module1.py", "module2.py"],
            "dependencies": []
        }
        
        # Mock auth
        self.mock_auth = Mock(spec=HubAuth)
        self.mock_auth.get_headers.return_value = {"Authorization": "Bearer test-token"}
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('greenlang.hub.client.httpx.Client')
    def test_client_initialization(self, mock_client):
        """Test HubClient initialization"""
        client = HubClient(
            registry_url="https://test.registry.io",
            auth=self.mock_auth
        )
        
        self.assertEqual(client.registry_url, "https://test.registry.io")
        self.assertEqual(client.auth, self.mock_auth)
        self.assertIsNotNone(client.session)
        
        # Test default URL
        client2 = HubClient()
        self.assertEqual(client2.registry_url, HubClient.DEFAULT_REGISTRY_URL)
    
    @patch('greenlang.hub.client.httpx.Client')
    @patch('greenlang.hub.client.create_pack_archive')
    @patch('greenlang.hub.client.load_manifest')
    def test_push_pack(self, mock_load_manifest, mock_create_archive, mock_client):
        """Test pushing pack to registry"""
        # Setup mocks
        mock_manifest = PackManifest(**self.test_manifest)
        mock_load_manifest.return_value = mock_manifest
        
        archive_path = self.temp_path / "test.tar.gz"
        archive_path.write_bytes(b"test archive content")
        mock_create_archive.return_value = archive_path
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "pack-123", "url": "https://hub/pack-123"}
        mock_response.raise_for_status = MagicMock()
        mock_session.post.return_value = mock_response
        mock_client.return_value = mock_session
        
        # Test push
        client = HubClient(auth=self.mock_auth)
        client.session = mock_session
        
        result = client.push(
            self.temp_path,
            tags=["test", "example"],
            description="Test description"
        )
        
        self.assertEqual(result["id"], "pack-123")
        mock_session.post.assert_called_once()
        
        # Verify request data
        call_args = mock_session.post.call_args
        self.assertEqual(call_args[0][0], "/api/v1/packs")
        self.assertIn("files", call_args[1])
        self.assertIn("data", call_args[1])
    
    @patch('greenlang.hub.client.httpx.Client')
    @patch('greenlang.hub.client.extract_pack_archive')
    def test_pull_pack(self, mock_extract, mock_client):
        """Test pulling pack from registry"""
        # Setup mocks
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": None,
            "manifest": self.test_manifest,
            "download_url": "https://hub/download/pack-123"
        }
        mock_response.raise_for_status = MagicMock()
        
        # Mock download response
        mock_download_response = MagicMock()
        mock_download_response.content = b"pack archive content"
        mock_download_response.raise_for_status = MagicMock()
        
        mock_session.get.side_effect = [mock_response, mock_download_response]
        mock_client.return_value = mock_session
        
        # Test pull
        client = HubClient()
        client.session = mock_session
        
        output_dir = self.temp_path / "output"
        result = client.pull("test-pack@1.0.0", output_dir, verify_signature=False)
        
        mock_session.get.assert_called()
        mock_extract.assert_called_once()
        
        # Verify calls
        calls = mock_session.get.call_args_list
        self.assertEqual(calls[0][0][0], "/api/v1/packs/test-pack@1.0.0")
    
    @patch('greenlang.hub.client.httpx.Client')
    def test_search_packs(self, mock_client):
        """Test searching for packs"""
        # Setup mocks
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"name": "pack1", "version": "1.0.0"},
            {"name": "pack2", "version": "2.0.0"}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_client.return_value = mock_session
        
        # Test search
        client = HubClient()
        client.session = mock_session
        
        results = client.search(
            query="test",
            tags=["example"],
            limit=10
        )
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["name"], "pack1")
        
        # Verify request params
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        self.assertEqual(call_args[0][0], "/api/v1/packs/search")
        self.assertIn("params", call_args[1])
        self.assertEqual(call_args[1]["params"]["q"], "test")
    
    @patch('greenlang.hub.client.httpx.Client')
    def test_list_packs(self, mock_client):
        """Test listing packs"""
        # Setup mocks
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"name": "pack1", "downloads": 100},
            {"name": "pack2", "downloads": 200}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_client.return_value = mock_session
        
        # Test list
        client = HubClient()
        client.session = mock_session
        
        packs = client.list_packs(user="testuser", limit=50)
        
        self.assertEqual(len(packs), 2)
        self.assertEqual(packs[1]["downloads"], 200)
        
        # Verify request
        mock_session.get.assert_called_once_with(
            "/api/v1/packs",
            params={"limit": 50, "offset": 0, "user": "testuser"}
        )
    
    @patch('greenlang.hub.client.httpx.Client')
    def test_delete_pack(self, mock_client):
        """Test deleting pack"""
        # Setup mocks
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_session.delete.return_value = mock_response
        mock_client.return_value = mock_session
        
        # Test delete without auth (should fail)
        client = HubClient()
        client.session = mock_session
        
        with self.assertRaises(ValueError):
            client.delete_pack("test-pack")
        
        # Test delete with auth
        client_auth = HubClient(auth=self.mock_auth)
        client_auth.session = mock_session
        
        result = client_auth.delete_pack("test-pack@1.0.0")
        
        self.assertTrue(result)
        mock_session.delete.assert_called_once_with("/api/v1/packs/test-pack@1.0.0")


class TestPackArchive(unittest.TestCase):
    """Test pack archive functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test pack structure
        self.pack_dir = self.temp_path / "test-pack"
        self.pack_dir.mkdir()
        
        (self.pack_dir / "module1.py").write_text("print('module1')")
        (self.pack_dir / "module2.py").write_text("print('module2')")
        (self.pack_dir / "manifest.json").write_text(json.dumps({
            "name": "test-pack",
            "version": "1.0.0",
            "description": "Test pack"
        }))
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_archive(self):
        """Test creating pack archive"""
        archive_path = create_pack_archive(self.pack_dir)
        
        self.assertTrue(archive_path.exists())
        self.assertTrue(archive_path.name.endswith('.tar.gz'))
        
        # Verify archive content
        import tarfile
        with tarfile.open(archive_path, 'r:gz') as tar:
            members = tar.getnames()
            self.assertIn('test-pack/module1.py', members)
            self.assertIn('test-pack/module2.py', members)
            self.assertIn('test-pack/manifest.json', members)
    
    def test_extract_archive(self):
        """Test extracting pack archive"""
        # Create archive
        archive_path = create_pack_archive(self.pack_dir)
        
        # Extract to new location
        extract_dir = self.temp_path / "extracted"
        result_dir = extract_pack_archive(archive_path, extract_dir)
        
        self.assertTrue(result_dir.exists())
        self.assertTrue((result_dir / "module1.py").exists())
        self.assertTrue((result_dir / "module2.py").exists())
        self.assertTrue((result_dir / "manifest.json").exists())
    
    def test_archive_exclusions(self):
        """Test that certain files are excluded from archive"""
        # Add files that should be excluded
        (self.pack_dir / "__pycache__").mkdir()
        (self.pack_dir / "__pycache__" / "test.pyc").write_text("pyc content")
        (self.pack_dir / ".git").mkdir()
        (self.pack_dir / ".git" / "config").write_text("git config")
        
        archive_path = create_pack_archive(self.pack_dir)
        
        # Verify exclusions
        import tarfile
        with tarfile.open(archive_path, 'r:gz') as tar:
            members = tar.getnames()
            self.assertNotIn('test-pack/__pycache__', members)
            self.assertNotIn('test-pack/.git', members)


class TestPackManifest(unittest.TestCase):
    """Test manifest handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.valid_manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "description": "Test pack description"
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_manifest(self):
        """Test creating new manifest"""
        # Create test pack directory with files
        pack_dir = self.temp_path / "mypack"
        pack_dir.mkdir()
        (pack_dir / "main.py").write_text("# main module")
        (pack_dir / "utils.py").write_text("# utils")
        (pack_dir / "data.json").write_text("{}")
        
        manifest = create_manifest(
            pack_dir,
            name="mypack",
            version="0.1.0",
            description="My test pack",
            author="Test Author"
        )
        
        self.assertEqual(manifest.name, "mypack")
        self.assertEqual(manifest.version, "0.1.0")
        self.assertIn("main.py", manifest.modules)
        self.assertIn("utils.py", manifest.modules)
        self.assertIn("data.json", manifest.resources)
    
    def test_load_save_manifest(self):
        """Test loading and saving manifest"""
        from greenlang.hub.manifest import save_manifest
        
        # Save manifest
        manifest = PackManifest(**self.valid_manifest)
        manifest_file = save_manifest(self.temp_path, manifest, format='json')
        
        self.assertTrue(manifest_file.exists())
        
        # Load manifest
        loaded = load_manifest(self.temp_path)
        
        self.assertEqual(loaded.name, manifest.name)
        self.assertEqual(loaded.version, manifest.version)
        self.assertEqual(loaded.description, manifest.description)
    
    def test_manifest_validation(self):
        """Test manifest validation"""
        # Valid manifest
        valid = PackManifest(**self.valid_manifest)
        self.assertEqual(valid.name, "test-pack")
        
        # Invalid version format
        with self.assertRaises(ValueError):
            PackManifest(
                name="test",
                version="invalid-version",
                description="Test"
            )
        
        # Invalid name format
        with self.assertRaises(ValueError):
            PackManifest(
                name="123-invalid",
                version="1.0.0",
                description="Test"
            )


class TestHubAuth(unittest.TestCase):
    """Test authentication functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_auth_headers(self):
        """Test getting authentication headers"""
        # Test with token
        auth_token = HubAuth(token="test-token")
        headers = auth_token.get_headers()
        self.assertEqual(headers["Authorization"], "Bearer test-token")
        
        # Test with API key
        auth_api = HubAuth(api_key="test-api-key")
        headers = auth_api.get_headers()
        self.assertEqual(headers["X-API-Key"], "test-api-key")
        
        # Test with username
        auth_user = HubAuth(username="testuser", token="token")
        headers = auth_user.get_headers()
        self.assertEqual(headers["X-Username"], "testuser")
    
    @patch('greenlang.hub.auth.httpx.post')
    def test_login(self, mock_post):
        """Test login functionality"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"token": "new-token"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        auth = HubAuth()
        result = auth.login("testuser", "testpass")
        
        self.assertTrue(result)
        self.assertEqual(auth.token, "new-token")
        self.assertEqual(auth.username, "testuser")
    
    def test_pack_signer(self):
        """Test pack signing functionality"""
        signer = PackSigner()
        
        # Generate keys
        private_key, public_key = signer.generate_keys(2048)
        self.assertIsNotNone(private_key)
        self.assertIsNotNone(public_key)
        
        # Sign data
        test_data = b"test pack content"
        signature = signer.sign_pack(test_data)
        
        self.assertIn("signature", signature)
        self.assertIn("hash", signature)
        self.assertIn("algorithm", signature)
        
        # Verify signature
        is_valid = signer.verify_signature(test_data, signature)
        self.assertTrue(is_valid)
        
        # Verify with wrong data should fail
        wrong_data = b"wrong content"
        is_valid = signer.verify_signature(wrong_data, signature)
        self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()