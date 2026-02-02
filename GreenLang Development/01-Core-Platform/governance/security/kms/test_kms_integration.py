"""
Test KMS Integration
====================

Comprehensive tests for KMS provider implementations.
Run with: pytest greenlang/security/kms/test_kms_integration.py -v
"""

import os
import pytest
import hashlib
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from greenlang.security.kms.base_kms import (
    BaseKMSProvider,
    KMSConfig,
    KMSSignResult,
    KMSKeyInfo,
    KMSProviderError,
    KMSKeyNotFoundError,
    KMSSigningError,
    KeyAlgorithm,
    SigningAlgorithm,
    KeyCache,
)
from greenlang.security.kms.factory import (
    detect_kms_provider,
    detect_provider_from_key_id,
    create_kms_provider,
    create_config_from_env,
    list_available_providers,
    get_provider_requirements,
)


class TestKeyCache:
    """Test key caching functionality"""

    def test_cache_basic_operations(self):
        """Test basic cache operations"""
        cache = KeyCache(ttl_seconds=5)

        # Create test key info
        key_info = KMSKeyInfo(
            key_id="test-key",
            key_arn="arn:aws:kms:us-east-1:123456789012:key/test-key",
            algorithm=KeyAlgorithm.RSA_2048,
            created_at=datetime.utcnow(),
            enabled=True,
            rotation_enabled=False,
        )

        # Test set and get
        cache.set("test-key", key_info)
        retrieved = cache.get("test-key")
        assert retrieved is not None
        assert retrieved.key_id == "test-key"

        # Test cache miss
        assert cache.get("non-existent") is None

    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        import time

        cache = KeyCache(ttl_seconds=1)

        key_info = KMSKeyInfo(
            key_id="test-key",
            key_arn="test-arn",
            algorithm=KeyAlgorithm.RSA_2048,
            created_at=datetime.utcnow(),
            enabled=True,
            rotation_enabled=False,
        )

        cache.set("test-key", key_info)
        assert cache.get("test-key") is not None

        # Wait for expiration
        time.sleep(1.5)
        assert cache.get("test-key") is None

    def test_cache_invalidation(self):
        """Test cache invalidation"""
        cache = KeyCache()

        key_info1 = KMSKeyInfo(
            key_id="key1",
            key_arn="arn1",
            algorithm=KeyAlgorithm.RSA_2048,
            created_at=datetime.utcnow(),
            enabled=True,
            rotation_enabled=False,
        )

        key_info2 = KMSKeyInfo(
            key_id="key2",
            key_arn="arn2",
            algorithm=KeyAlgorithm.ECDSA_P256,
            created_at=datetime.utcnow(),
            enabled=True,
            rotation_enabled=False,
        )

        cache.set("key1", key_info1)
        cache.set("key2", key_info2)

        # Invalidate specific key
        cache.invalidate("key1")
        assert cache.get("key1") is None
        assert cache.get("key2") is not None

        # Invalidate all
        cache.invalidate()
        assert cache.get("key2") is None


class TestProviderDetection:
    """Test automatic provider detection"""

    def test_detect_aws_from_env(self):
        """Test AWS detection from environment"""
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test-key"}):
            assert detect_kms_provider() == "aws"

        with patch.dict(os.environ, {"AWS_PROFILE": "test-profile"}):
            assert detect_kms_provider() == "aws"

    def test_detect_azure_from_env(self):
        """Test Azure detection from environment"""
        with patch.dict(os.environ, {"AZURE_CLIENT_ID": "test-client"}):
            assert detect_kms_provider() == "azure"

        with patch.dict(os.environ, {"MSI_ENDPOINT": "http://test"}):
            assert detect_kms_provider() == "azure"

    def test_detect_gcp_from_env(self):
        """Test GCP detection from environment"""
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/key.json"}):
            assert detect_kms_provider() == "gcp"

        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}):
            assert detect_kms_provider() == "gcp"

    def test_detect_from_key_id(self):
        """Test provider detection from key ID format"""
        # AWS ARN
        assert detect_provider_from_key_id(
            "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
        ) == "aws"

        # Azure Key Vault URL
        assert detect_provider_from_key_id(
            "https://myvault.vault.azure.net/keys/mykey"
        ) == "azure"

        # GCP resource name
        assert detect_provider_from_key_id(
            "projects/my-project/locations/global/keyRings/my-ring/cryptoKeys/my-key"
        ) == "gcp"

        # Unknown format
        assert detect_provider_from_key_id("some-random-key-id") is None

    def test_config_from_env(self):
        """Test configuration creation from environment"""
        env_vars = {
            "GL_KMS_PROVIDER": "aws",
            "GL_KMS_KEY_ID": "test-key",
            "GL_KMS_REGION": "us-west-2",
            "GL_KMS_CACHE_TTL": "600",
            "GL_KMS_MAX_RETRIES": "5",
            "GL_KMS_TIMEOUT": "60",
            "GL_KMS_ASYNC_ENABLED": "false",
            "AWS_PROFILE": "test-profile",
        }

        with patch.dict(os.environ, env_vars):
            config = create_config_from_env()

            assert config.provider == "aws"
            assert config.key_id == "test-key"
            assert config.region == "us-west-2"
            assert config.cache_ttl_seconds == 600
            assert config.max_retries == 5
            assert config.timeout_seconds == 60
            assert config.async_enabled is False
            assert config.aws_profile == "test-profile"


class TestAWSKMSProvider:
    """Test AWS KMS provider implementation"""

    @pytest.fixture
    def mock_boto3(self):
        """Mock boto3 for testing"""
        with patch("greenlang.security.kms.aws_kms.boto3") as mock:
            yield mock

    @pytest.fixture
    def aws_config(self):
        """AWS KMS configuration"""
        return KMSConfig(
            provider="aws",
            key_id="arn:aws:kms:us-east-1:123456789012:key/test-key",
            region="us-east-1",
            cache_ttl_seconds=300,
        )

    def test_aws_key_info(self, mock_boto3, aws_config):
        """Test getting AWS key information"""
        from greenlang.security.kms.aws_kms import AWSKMSProvider

        # Mock client
        mock_client = Mock()
        mock_session = Mock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session

        # Mock describe_key response
        mock_client.describe_key.return_value = {
            "KeyMetadata": {
                "KeyId": "12345678-1234-1234-1234-123456789012",
                "Arn": "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012",
                "KeyUsage": "SIGN_VERIFY",
                "KeySpec": "RSA_2048",
                "Enabled": True,
                "CreationDate": datetime.utcnow(),
                "KeyState": "Enabled",
                "KeyRotationEnabled": False,
            }
        }

        # Create provider and get key info
        provider = AWSKMSProvider(aws_config)
        key_info = provider.get_key_info()

        assert key_info.key_id == "12345678-1234-1234-1234-123456789012"
        assert key_info.algorithm == KeyAlgorithm.RSA_2048
        assert key_info.enabled is True
        assert key_info.rotation_enabled is False

    def test_aws_signing(self, mock_boto3, aws_config):
        """Test AWS KMS signing"""
        from greenlang.security.kms.aws_kms import AWSKMSProvider

        # Mock client
        mock_client = Mock()
        mock_session = Mock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session

        # Mock responses
        mock_client.describe_key.return_value = {
            "KeyMetadata": {
                "KeyId": "test-key",
                "Arn": "arn:aws:kms:us-east-1:123456789012:key/test-key",
                "Enabled": True,
                "CreationDate": datetime.utcnow(),
            }
        }

        mock_client.sign.return_value = {
            "KeyId": "test-key",
            "Signature": b"mock-signature",
            "SigningAlgorithm": "RSASSA_PSS_SHA_256",
        }

        # Create provider and sign
        provider = AWSKMSProvider(aws_config)
        data = b"test data"
        result = provider.sign(data)

        assert result["signature"] == b"mock-signature"
        assert result["key_id"] == "test-key"
        assert result["algorithm"] == "RSASSA_PSS_SHA_256"
        assert result["provider"] == "aws"

        # Verify sign was called with correct parameters
        mock_client.sign.assert_called_once()
        call_args = mock_client.sign.call_args[1]
        assert call_args["KeyId"] == aws_config.key_id
        assert call_args["MessageType"] == "DIGEST"
        assert call_args["Message"] == hashlib.sha256(data).digest()


class TestAzureKeyVaultProvider:
    """Test Azure Key Vault provider implementation"""

    @pytest.fixture
    def mock_azure(self):
        """Mock Azure SDK for testing"""
        with patch("greenlang.security.kms.azure_kms.KeyClient") as mock_client, \
             patch("greenlang.security.kms.azure_kms.DefaultAzureCredential") as mock_cred, \
             patch("greenlang.security.kms.azure_kms.CryptographyClient") as mock_crypto:
            yield mock_client, mock_cred, mock_crypto

    @pytest.fixture
    def azure_config(self):
        """Azure Key Vault configuration"""
        return KMSConfig(
            provider="azure",
            key_id="https://myvault.vault.azure.net/keys/signing-key",
            azure_vault_url="https://myvault.vault.azure.net",
        )

    def test_azure_url_parsing(self):
        """Test Azure Key Vault URL parsing"""
        from greenlang.security.kms.azure_kms import AzureKeyVaultProvider

        config = KMSConfig(
            provider="azure",
            key_id="https://myvault.vault.azure.net/keys/signing-key",
        )

        # Mock Azure imports
        with patch("greenlang.security.kms.azure_kms.AZURE_SDK_AVAILABLE", True):
            with patch.object(AzureKeyVaultProvider, "_create_client"):
                provider = AzureKeyVaultProvider(config)

                assert provider.vault_url == "https://myvault.vault.azure.net"
                assert provider.key_name == "signing-key"


class TestGCPCloudKMSProvider:
    """Test GCP Cloud KMS provider implementation"""

    @pytest.fixture
    def mock_gcp(self):
        """Mock GCP SDK for testing"""
        with patch("greenlang.security.kms.gcp_kms.kms") as mock_kms:
            yield mock_kms

    @pytest.fixture
    def gcp_config(self):
        """GCP KMS configuration"""
        return KMSConfig(
            provider="gcp",
            key_id="signing-key",
            gcp_project_id="test-project",
            gcp_location_id="global",
            gcp_keyring_id="test-keyring",
        )

    def test_gcp_resource_name_building(self):
        """Test GCP resource name construction"""
        from greenlang.security.kms.gcp_kms import GCPCloudKMSProvider

        config = KMSConfig(
            provider="gcp",
            key_id="signing-key",
            gcp_project_id="my-project",
            gcp_location_id="us-central1",
            gcp_keyring_id="my-keyring",
        )

        # Mock GCP imports
        with patch("greenlang.security.kms.gcp_kms.GCP_SDK_AVAILABLE", True):
            with patch.object(GCPCloudKMSProvider, "_create_client"):
                provider = GCPCloudKMSProvider(config)

                resource_name = provider._get_key_resource_name()
                assert resource_name == (
                    "projects/my-project/locations/us-central1/"
                    "keyRings/my-keyring/cryptoKeys/signing-key"
                )

                version_name = provider._get_key_version_resource_name("signing-key", "3")
                assert version_name.endswith("/cryptoKeyVersions/3")


class TestRetryLogic:
    """Test retry logic with exponential backoff"""

    def test_retry_success_on_second_attempt(self):
        """Test successful retry on second attempt"""
        from greenlang.security.kms.base_kms import BaseKMSProvider

        # Create a mock provider
        config = KMSConfig(
            provider="test",
            key_id="test-key",
            max_retries=3,
            retry_base_delay=0.1,
        )

        class TestProvider(BaseKMSProvider):
            def _create_client(self):
                return Mock()

            def _create_async_client(self):
                return None

            def get_key_info(self, key_id=None):
                pass

            def sign(self, data, key_id=None, algorithm=None):
                pass

            def verify(self, data, signature, key_id=None, algorithm=None):
                pass

            def rotate_key(self, key_id=None):
                pass

            async def sign_async(self, data, key_id=None, algorithm=None):
                pass

        provider = TestProvider(config)

        # Mock function that fails once then succeeds
        call_count = 0

        def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return "success"

        result = provider.retry_with_backoff(mock_func)
        assert result == "success"
        assert call_count == 2

    def test_retry_all_attempts_fail(self):
        """Test all retry attempts fail"""
        from greenlang.security.kms.base_kms import BaseKMSProvider

        config = KMSConfig(
            provider="test",
            key_id="test-key",
            max_retries=2,
            retry_base_delay=0.01,
        )

        class TestProvider(BaseKMSProvider):
            def _create_client(self):
                return Mock()

            def _create_async_client(self):
                return None

            def get_key_info(self, key_id=None):
                pass

            def sign(self, data, key_id=None, algorithm=None):
                pass

            def verify(self, data, signature, key_id=None, algorithm=None):
                pass

            def rotate_key(self, key_id=None):
                pass

            async def sign_async(self, data, key_id=None, algorithm=None):
                pass

        provider = TestProvider(config)

        def mock_func():
            raise Exception("Permanent failure")

        with pytest.raises(Exception) as exc_info:
            provider.retry_with_backoff(mock_func)

        assert "Permanent failure" in str(exc_info.value)


class TestBatchOperations:
    """Test batch signing operations"""

    def test_batch_signing(self):
        """Test batch signing multiple documents"""
        from greenlang.security.kms.base_kms import BaseKMSProvider

        config = KMSConfig(
            provider="test",
            key_id="test-key",
            batch_size=2,
        )

        class TestProvider(BaseKMSProvider):
            def _create_client(self):
                return Mock()

            def _create_async_client(self):
                return None

            def get_key_info(self, key_id=None):
                pass

            def sign(self, data, key_id=None, algorithm=None):
                # Return mock signature
                return {
                    "signature": b"signed:" + data,
                    "key_id": key_id or self.config.key_id,
                    "algorithm": "RSA",
                    "timestamp": datetime.utcnow().isoformat(),
                    "key_version": "1",
                    "provider": "test",
                }

            def verify(self, data, signature, key_id=None, algorithm=None):
                pass

            def rotate_key(self, key_id=None):
                pass

            async def sign_async(self, data, key_id=None, algorithm=None):
                return self.sign(data, key_id, algorithm)

        provider = TestProvider(config)

        # Sign batch
        data_items = [b"doc1", b"doc2", b"doc3"]
        results = provider.sign_batch(data_items)

        assert len(results) == 3
        assert results[0]["signature"] == b"signed:doc1"
        assert results[1]["signature"] == b"signed:doc2"
        assert results[2]["signature"] == b"signed:doc3"


def test_provider_requirements():
    """Test getting provider requirements"""
    aws_reqs = get_provider_requirements("aws")
    assert "boto3" in aws_reqs

    azure_reqs = get_provider_requirements("azure")
    assert "azure-keyvault-keys" in azure_reqs

    gcp_reqs = get_provider_requirements("gcp")
    assert "google-cloud-kms" in gcp_reqs


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])