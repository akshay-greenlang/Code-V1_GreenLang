"""
Integration tests for the native OCI registry client.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from greenlang.registry.oci_client import (
    OCIClient, OCIAuth, OCIManifest, OCIDescriptor,
    create_client
)


class TestOCIAuth(unittest.TestCase):
    """Test OCI authentication mechanisms"""
    
    def test_basic_auth(self):
        """Test basic authentication header generation"""
        auth = OCIAuth(username="user", password="pass")
        header = auth.get_auth_header("test.registry.io")
        
        self.assertIsNotNone(header)
        self.assertTrue(header.startswith("Basic "))
        
        # Verify base64 encoding
        import base64
        expected = base64.b64encode(b"user:pass").decode()
        self.assertEqual(header, f"Basic {expected}")
    
    def test_bearer_token_auth(self):
        """Test bearer token authentication"""
        auth = OCIAuth(token="test-token-123")
        header = auth.get_auth_header("test.registry.io")
        
        self.assertEqual(header, "Bearer test-token-123")
    
    def test_www_authenticate_parsing(self):
        """Test parsing of WWW-Authenticate header"""
        auth = OCIAuth()
        header = 'Bearer realm="https://auth.docker.io/token",service="registry.docker.io",scope="repository:library/alpine:pull"'
        
        params = auth._parse_www_authenticate(header)
        
        self.assertEqual(params["realm"], "https://auth.docker.io/token")
        self.assertEqual(params["service"], "registry.docker.io")
        self.assertEqual(params["scope"], "repository:library/alpine:pull")
    
    @patch('urllib.request.urlopen')
    def test_bearer_token_request(self, mock_urlopen):
        """Test requesting bearer token from auth endpoint"""
        auth = OCIAuth(username="user", password="pass")
        
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"token": "new-token-456"}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        token = auth._request_bearer_token(
            "https://auth.registry.io/token",
            "registry.io",
            "repository:test/repo:push"
        )
        
        self.assertEqual(token, "new-token-456")


class TestOCIManifest(unittest.TestCase):
    """Test OCI manifest handling"""
    
    def test_manifest_creation(self):
        """Test creating an OCI manifest"""
        manifest = OCIManifest(
            config={
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": "sha256:abc123",
                "size": 1024
            },
            layers=[
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": "sha256:def456",
                    "size": 2048
                }
            ],
            annotations={"org.opencontainers.image.title": "Test Pack"}
        )
        
        manifest_json = manifest.to_json()
        data = json.loads(manifest_json)
        
        self.assertEqual(data["schemaVersion"], 2)
        self.assertEqual(data["config"]["digest"], "sha256:abc123")
        self.assertEqual(len(data["layers"]), 1)
        self.assertEqual(data["annotations"]["org.opencontainers.image.title"], "Test Pack")
    
    def test_manifest_from_json(self):
        """Test parsing manifest from JSON"""
        manifest_data = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": "sha256:xyz789",
                "size": 512
            },
            "layers": [],
            "annotations": {"test": "value"}
        }
        
        manifest = OCIManifest.from_json(json.dumps(manifest_data))
        
        self.assertEqual(manifest.schema_version, 2)
        self.assertEqual(manifest.config["digest"], "sha256:xyz789")
        self.assertEqual(manifest.annotations["test"], "value")


class TestOCIClient(unittest.TestCase):
    """Test OCI client operations"""
    
    def setUp(self):
        self.client = OCIClient(registry="test.registry.io")
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_calculate_digest(self):
        """Test SHA256 digest calculation"""
        data = b"test data"
        digest = self.client._calculate_digest(data)
        
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(len(digest), 71)  # "sha256:" + 64 hex chars
    
    @patch.object(OCIClient, '_make_request')
    def test_push_blob_existing(self, mock_request):
        """Test pushing a blob that already exists"""
        # Mock HEAD request returns 200 (blob exists)
        mock_request.return_value = (200, b"", {})
        
        data = b"test blob data"
        descriptor = self.client.push_blob("test", "repo", data)
        
        self.assertEqual(descriptor.size, len(data))
        self.assertTrue(descriptor.digest.startswith("sha256:"))
        
        # Should only make HEAD request, not upload
        self.assertEqual(mock_request.call_count, 1)
    
    @patch.object(OCIClient, '_make_request')
    def test_push_blob_new(self, mock_request):
        """Test pushing a new blob"""
        # Mock responses for upload flow
        mock_request.side_effect = [
            (404, b"", {}),  # HEAD - blob doesn't exist
            (202, b"", {"Location": "/v2/test/repo/blobs/uploads/uuid"}),  # POST - initiate
            (201, b"", {})   # PUT - complete upload
        ]
        
        data = b"test blob data"
        descriptor = self.client.push_blob("test", "repo", data)
        
        self.assertEqual(descriptor.size, len(data))
        self.assertEqual(mock_request.call_count, 3)
    
    @patch.object(OCIClient, '_make_request')
    def test_pull_blob(self, mock_request):
        """Test pulling a blob"""
        expected_data = b"pulled blob data"
        mock_request.return_value = (200, expected_data, {})
        
        data = self.client.pull_blob("test", "repo", "sha256:abc123")
        
        self.assertEqual(data, expected_data)
        mock_request.assert_called_once()
    
    @patch.object(OCIClient, '_make_request')
    def test_push_manifest(self, mock_request):
        """Test pushing a manifest"""
        mock_request.return_value = (201, b"", {})
        
        manifest = OCIManifest(
            config={"digest": "sha256:config", "size": 100},
            layers=[{"digest": "sha256:layer", "size": 200}]
        )
        
        digest = self.client.push_manifest("test", "repo", "v1.0.0", manifest)
        
        self.assertTrue(digest.startswith("sha256:"))
        mock_request.assert_called_once()
    
    @patch.object(OCIClient, '_make_request')
    def test_pull_manifest(self, mock_request):
        """Test pulling a manifest"""
        manifest_data = {
            "schemaVersion": 2,
            "config": {"digest": "sha256:config"},
            "layers": []
        }
        mock_request.return_value = (200, json.dumps(manifest_data).encode(), {})
        
        manifest = self.client.pull_manifest("test", "repo", "v1.0.0")
        
        self.assertEqual(manifest.config["digest"], "sha256:config")
        mock_request.assert_called_once()
    
    @patch.object(OCIClient, '_make_request')
    def test_list_tags(self, mock_request):
        """Test listing repository tags"""
        response_data = {"tags": ["v1.0.0", "v1.1.0", "latest"]}
        mock_request.return_value = (200, json.dumps(response_data).encode(), {})
        
        tags = self.client.list_tags("test", "repo")
        
        self.assertEqual(len(tags), 3)
        self.assertIn("v1.0.0", tags)
        self.assertIn("latest", tags)
    
    @patch.object(OCIClient, 'push_blob')
    @patch.object(OCIClient, 'push_manifest')
    def test_push_pack(self, mock_push_manifest, mock_push_blob):
        """Test pushing a GreenLang pack"""
        # Create test pack directory
        pack_dir = Path(self.temp_dir) / "test-pack"
        pack_dir.mkdir()
        
        manifest_file = pack_dir / "pack.yaml"
        manifest_file.write_text("name: test-pack\nversion: 1.0.0")
        
        # Mock blob push
        mock_push_blob.return_value = OCIDescriptor(
            media_type="application/vnd.greenlang.pack.v1.tar+gzip",
            digest="sha256:pack123",
            size=1024
        )
        
        # Mock manifest push
        mock_push_manifest.return_value = "sha256:manifest456"
        
        digest = self.client.push_pack("org", "test-pack", "1.0.0", pack_dir)
        
        self.assertEqual(digest, "sha256:manifest456")
        self.assertEqual(mock_push_blob.call_count, 2)  # Pack + config
        mock_push_manifest.assert_called_once()
    
    @patch.object(OCIClient, 'pull_manifest')
    @patch.object(OCIClient, 'pull_blob')
    def test_pull_pack(self, mock_pull_blob, mock_pull_manifest):
        """Test pulling a GreenLang pack"""
        # Mock manifest
        mock_pull_manifest.return_value = OCIManifest(
            layers=[{
                "mediaType": "application/vnd.greenlang.pack.v1.tar+gzip",
                "digest": "sha256:pack789",
                "size": 2048
            }]
        )
        
        # Create test pack archive
        import tarfile
        import io
        
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Add a test file
            info = tarfile.TarInfo(name="pack.yaml")
            info.size = 20
            tar.addfile(info, io.BytesIO(b"name: pulled-pack\n"))
        
        mock_pull_blob.return_value = tar_buffer.getvalue()
        
        output_dir = Path(self.temp_dir) / "output"
        pack_dir = self.client.pull_pack("org", "test-pack", "1.0.0", output_dir)
        
        self.assertTrue(pack_dir.exists())
        self.assertTrue((pack_dir / "pack.yaml").exists())


class TestRegistryIntegration(unittest.TestCase):
    """Integration tests for registry operations"""
    
    @patch('urllib.request.urlopen')
    def test_authentication_flow(self, mock_urlopen):
        """Test complete authentication flow"""
        client = create_client(
            registry="test.registry.io",
            username="user",
            password="pass"
        )
        
        # Mock initial 401 response with WWW-Authenticate
        auth_response = MagicMock()
        auth_response.code = 401
        auth_response.headers = {
            "WWW-Authenticate": 'Bearer realm="https://auth.test.io/token",service="test.registry.io"'
        }
        
        # Mock token request response
        token_response = MagicMock()
        token_response.read.return_value = json.dumps({"token": "test-token"}).encode()
        
        # Mock authenticated request
        success_response = MagicMock()
        success_response.code = 200
        success_response.read.return_value = b'{"tags": ["v1.0.0"]}'
        success_response.headers = {}
        
        # Set up mock sequence
        mock_urlopen.side_effect = [
            auth_response,
            token_response.__enter__(),
            success_response.__enter__()
        ]
        
        # This should trigger authentication flow
        code, data, headers = client._make_request(
            "GET",
            "https://test.registry.io/v2/test/repo/tags/list"
        )
        
        self.assertEqual(code, 200)
        self.assertIn(b"v1.0.0", data)


class TestDependencyResolver(unittest.TestCase):
    """Test dependency resolver"""
    
    def setUp(self):
        # Import dependency resolver
        from greenlang.packs.dependency_resolver import (
            DependencyResolver, DependencySpec, PackageInfo,
            ResolutionStrategy
        )
        
        self.DependencyResolver = DependencyResolver
        self.DependencySpec = DependencySpec
        self.PackageInfo = PackageInfo
        self.ResolutionStrategy = ResolutionStrategy
    
    def test_simple_resolution(self):
        """Test simple dependency resolution"""
        resolver = self.DependencyResolver(strategy=self.ResolutionStrategy.LATEST)
        
        # Add packages to pool
        resolver.add_package_to_pool(self.PackageInfo(
            name="package-a",
            version="1.0.0",
            dependencies=[]
        ))
        
        resolver.add_package_to_pool(self.PackageInfo(
            name="package-a",
            version="2.0.0",
            dependencies=[]
        ))
        
        # Resolve
        requirements = [self.DependencySpec(name="package-a", version_spec=">=1.0.0")]
        result = resolver.resolve(requirements)
        
        self.assertTrue(result.success)
        self.assertEqual(result.resolved["package-a"], "2.0.0")  # Latest strategy
    
    def test_conflict_detection(self):
        """Test dependency conflict detection"""
        resolver = self.DependencyResolver()
        
        # Add packages with conflicting dependencies
        resolver.add_package_to_pool(self.PackageInfo(
            name="package-a",
            version="1.0.0",
            dependencies=[
                self.DependencySpec(name="package-c", version_spec="==1.0.0")
            ]
        ))
        
        resolver.add_package_to_pool(self.PackageInfo(
            name="package-b",
            version="1.0.0",
            dependencies=[
                self.DependencySpec(name="package-c", version_spec="==2.0.0")
            ]
        ))
        
        resolver.add_package_to_pool(self.PackageInfo(
            name="package-c",
            version="1.0.0",
            dependencies=[]
        ))
        
        resolver.add_package_to_pool(self.PackageInfo(
            name="package-c",
            version="2.0.0",
            dependencies=[]
        ))
        
        # Try to resolve conflicting requirements
        requirements = [
            self.DependencySpec(name="package-a", version_spec="==1.0.0"),
            self.DependencySpec(name="package-b", version_spec="==1.0.0")
        ]
        
        result = resolver.resolve(requirements)
        
        # Should detect conflict
        self.assertFalse(result.success)
        self.assertTrue(len(result.conflicts) > 0)
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        resolver = self.DependencyResolver()
        
        # Create circular dependency: A -> B -> C -> A
        resolver.add_package_to_pool(self.PackageInfo(
            name="package-a",
            version="1.0.0",
            dependencies=[
                self.DependencySpec(name="package-b", version_spec=">=1.0.0")
            ]
        ))
        
        resolver.add_package_to_pool(self.PackageInfo(
            name="package-b",
            version="1.0.0",
            dependencies=[
                self.DependencySpec(name="package-c", version_spec=">=1.0.0")
            ]
        ))
        
        resolver.add_package_to_pool(self.PackageInfo(
            name="package-c",
            version="1.0.0",
            dependencies=[
                self.DependencySpec(name="package-a", version_spec=">=1.0.0")
            ]
        ))
        
        requirements = [self.DependencySpec(name="package-a", version_spec=">=1.0.0")]
        result = resolver.resolve(requirements)
        
        # Should detect circular dependency
        self.assertTrue(len(result.warnings) > 0)
        circular_warning = any("Circular dependency" in w for w in result.warnings)
        self.assertTrue(circular_warning)


if __name__ == "__main__":
    unittest.main()