# KMS Integration Guide for GreenLang

## Overview

GreenLang now supports comprehensive Key Management Service (KMS) integration with AWS KMS, Azure Key Vault, and Google Cloud KMS. This guide provides setup instructions and usage examples for each provider.

## Features

- **Multi-Cloud Support**: AWS KMS, Azure Key Vault, Google Cloud KMS
- **Automatic Provider Detection**: Detects cloud environment automatically
- **Key Caching**: Reduces API calls with configurable TTL (default: 5 minutes)
- **Retry Logic**: Exponential backoff for transient failures
- **Async Support**: Batch and async signing operations
- **Key Rotation**: Support for automatic and manual key rotation
- **Algorithm Support**: RSA (2048/3072/4096), ECDSA (P-256/P-384/P-521)

## Installation

### AWS KMS
```bash
pip install boto3
# Optional for async support
pip install aioboto3
```

### Azure Key Vault
```bash
pip install azure-keyvault-keys azure-identity
# Optional for async support
pip install azure-keyvault-keys[aio]
```

### Google Cloud KMS
```bash
pip install google-cloud-kms crc32c
```

## Configuration

### Environment Variables

```bash
# General KMS Configuration
export GL_KMS_PROVIDER=aws|azure|gcp  # Optional if auto-detection works
export GL_KMS_KEY_ID=<your-key-id>
export GL_KMS_CACHE_TTL=300           # Cache TTL in seconds
export GL_KMS_MAX_RETRIES=3           # Max retry attempts
export GL_KMS_TIMEOUT=30              # Timeout in seconds
export GL_KMS_ASYNC_ENABLED=true      # Enable async support

# For signing mode
export GL_SIGNING_MODE=kms            # Enable KMS signing
```

## Provider-Specific Setup

### AWS KMS Setup

1. **Create a KMS Key**:
```bash
aws kms create-key \
    --key-usage SIGN_VERIFY \
    --key-spec RSA_2048 \
    --description "GreenLang signing key"
```

2. **Configure Environment**:
```bash
# Using AWS CLI credentials
export AWS_PROFILE=your-profile
export AWS_REGION=us-east-1
export GL_KMS_KEY_ID=arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012

# Or using IAM role (EC2/Lambda)
# Credentials are automatically obtained from instance metadata
export GL_KMS_KEY_ID=alias/greenlang-signing-key
```

3. **IAM Policy Requirements**:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "kms:Sign",
                "kms:Verify",
                "kms:DescribeKey",
                "kms:GetPublicKey",
                "kms:ListKeys",
                "kms:GetKeyRotationStatus"
            ],
            "Resource": "arn:aws:kms:*:*:key/*"
        }
    ]
}
```

4. **Usage Example**:
```python
import os
from greenlang.security.signing import create_signer, SigningConfig

# Configure for AWS KMS
os.environ['GL_SIGNING_MODE'] = 'kms'
os.environ['GL_KMS_PROVIDER'] = 'aws'
os.environ['GL_KMS_KEY_ID'] = 'arn:aws:kms:us-east-1:123456789012:key/abc-123'
os.environ['AWS_REGION'] = 'us-east-1'

# Create signer
config = SigningConfig(mode='kms', kms_key_id=os.environ['GL_KMS_KEY_ID'])
signer = create_signer(config)

# Sign data
data = b"Important GreenLang data"
result = signer.sign(data)
print(f"Signature: {result['signature'].hex()}")
print(f"Algorithm: {result['algorithm']}")
```

### Azure Key Vault Setup

1. **Create a Key Vault and Key**:
```bash
# Create Key Vault
az keyvault create \
    --name greenlang-vault \
    --resource-group myResourceGroup \
    --location eastus

# Create signing key
az keyvault key create \
    --vault-name greenlang-vault \
    --name signing-key \
    --kty RSA \
    --size 2048 \
    --ops sign verify
```

2. **Configure Environment**:
```bash
# Using Service Principal
export AZURE_TENANT_ID=your-tenant-id
export AZURE_CLIENT_ID=your-client-id
export AZURE_CLIENT_SECRET=your-client-secret
export AZURE_KEY_VAULT_URL=https://greenlang-vault.vault.azure.net/
export GL_KMS_KEY_ID=signing-key

# Or using Managed Identity (Azure VMs)
# Identity is automatically obtained
export AZURE_KEY_VAULT_URL=https://greenlang-vault.vault.azure.net/
export GL_KMS_KEY_ID=signing-key
```

3. **Access Policy Requirements**:
```json
{
    "keys": [
        "get",
        "list",
        "sign",
        "verify",
        "wrapKey",
        "unwrapKey"
    ]
}
```

4. **Usage Example**:
```python
import os
from greenlang.security.signing import create_signer, SigningConfig

# Configure for Azure Key Vault
os.environ['GL_SIGNING_MODE'] = 'kms'
os.environ['GL_KMS_PROVIDER'] = 'azure'
os.environ['AZURE_KEY_VAULT_URL'] = 'https://your-vault.vault.azure.net/'
os.environ['GL_KMS_KEY_ID'] = 'signing-key'

# Create signer
config = SigningConfig(
    mode='kms',
    kms_key_id='signing-key'
)
signer = create_signer(config)

# Sign data
data = b"Critical sustainability metrics"
result = signer.sign(data)
print(f"Signed with Azure Key Vault: {result['timestamp']}")
```

### Google Cloud KMS Setup

1. **Create a Keyring and Key**:
```bash
# Create keyring
gcloud kms keyrings create greenlang-keyring \
    --location global

# Create signing key
gcloud kms keys create signing-key \
    --location global \
    --keyring greenlang-keyring \
    --purpose asymmetric-signing \
    --default-algorithm rsa-sign-pss-2048-sha256
```

2. **Configure Environment**:
```bash
# Using Service Account
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
export GCP_PROJECT_ID=your-project-id
export GCP_LOCATION=global
export GCP_KEYRING=greenlang-keyring
export GL_KMS_KEY_ID=signing-key

# Or using Application Default Credentials
gcloud auth application-default login
export GCP_PROJECT_ID=your-project-id
export GCP_LOCATION=global
export GCP_KEYRING=greenlang-keyring
export GL_KMS_KEY_ID=signing-key
```

3. **IAM Requirements**:
```yaml
bindings:
- role: roles/cloudkms.signerVerifier
  members:
  - serviceAccount:your-service-account@project.iam.gserviceaccount.com
- role: roles/cloudkms.viewer
  members:
  - serviceAccount:your-service-account@project.iam.gserviceaccount.com
```

4. **Usage Example**:
```python
import os
from greenlang.security.signing import create_signer, SigningConfig

# Configure for Google Cloud KMS
os.environ['GL_SIGNING_MODE'] = 'kms'
os.environ['GL_KMS_PROVIDER'] = 'gcp'
os.environ['GCP_PROJECT_ID'] = 'your-project'
os.environ['GCP_LOCATION'] = 'global'
os.environ['GCP_KEYRING'] = 'greenlang-keyring'
os.environ['GL_KMS_KEY_ID'] = 'signing-key'

# Create signer
config = SigningConfig(mode='kms')
signer = create_signer(config)

# Sign data
data = b"Environmental compliance report"
result = signer.sign(data)
print(f"Signed with GCP KMS: {result['key_id']}")
```

## Advanced Features

### Batch Signing

```python
from greenlang.security.kms import create_kms_provider, KMSConfig

# Create KMS provider
config = KMSConfig(
    provider='aws',
    key_id='your-key-id',
    batch_size=25
)
provider = create_kms_provider(config)

# Sign multiple items
data_items = [
    b"Document 1",
    b"Document 2",
    b"Document 3"
]
results = provider.sign_batch(data_items)
```

### Async Signing

```python
import asyncio
from greenlang.security.kms import create_kms_provider, KMSConfig

async def async_sign_example():
    config = KMSConfig(
        provider='aws',
        key_id='your-key-id',
        async_enabled=True
    )
    provider = create_kms_provider(config)

    # Async signing
    data = b"Async data"
    result = await provider.sign_async(data)
    return result

# Run async
result = asyncio.run(async_sign_example())
```

### Key Rotation

```python
from greenlang.security.kms import create_kms_provider

provider = create_kms_provider()

# Enable automatic rotation (AWS)
provider.rotate_key()

# Create new key version (Azure/GCP)
new_version = provider.rotate_key()
print(f"New key version: {new_version}")
```

### Signature Verification

```python
from greenlang.security.kms import create_kms_provider

provider = create_kms_provider()

# Sign data
data = b"Verify this"
sign_result = provider.sign(data)

# Verify signature
is_valid = provider.verify(
    data=data,
    signature=sign_result['signature']
)
print(f"Signature valid: {is_valid}")
```

## Performance Optimization

### Key Caching

The KMS integration includes automatic key caching to reduce API calls:

```python
# Configure cache TTL (default: 300 seconds)
os.environ['GL_KMS_CACHE_TTL'] = '600'  # 10 minutes

# Invalidate cache when needed
provider.invalidate_cache()  # Clear all
provider.invalidate_cache('specific-key-id')  # Clear specific key
```

### Connection Pooling

Each provider maintains connection pools for optimal performance:

- **AWS**: Uses boto3 session pooling
- **Azure**: Uses azure-core connection pooling
- **GCP**: Uses gRPC channel pooling

## Error Handling

```python
from greenlang.security.kms import (
    KMSProviderError,
    KMSKeyNotFoundError,
    KMSSigningError,
    KMSKeyRotationError
)

try:
    signer = create_signer(config)
    result = signer.sign(data)
except KMSKeyNotFoundError as e:
    print(f"Key not found: {e}")
except KMSSigningError as e:
    print(f"Signing failed: {e}")
except KMSProviderError as e:
    print(f"Provider error: {e}")
```

## Security Best Practices

1. **Use IAM/RBAC**: Never use root credentials or overly permissive policies
2. **Enable Key Rotation**: Rotate keys regularly (90 days recommended)
3. **Audit Logging**: Enable CloudTrail/Azure Monitor/Cloud Audit Logs
4. **Use HSM Keys**: For production, use HSM-backed keys when possible
5. **Least Privilege**: Grant minimum required permissions
6. **Network Security**: Use VPC endpoints/Private Link/Private Service Connect
7. **Monitoring**: Set up alerts for key usage anomalies

## Troubleshooting

### Auto-Detection Not Working

```python
from greenlang.security.kms import detect_kms_provider, list_available_providers

# Check which provider is detected
detected = detect_kms_provider()
print(f"Detected provider: {detected}")

# List available providers based on installed packages
available = list_available_providers()
print(f"Available providers: {available}")
```

### Missing Dependencies

```python
from greenlang.security.kms import get_provider_requirements

# Get requirements for a provider
requirements = get_provider_requirements('aws')
print(f"Install: pip install {' '.join(requirements)}")
```

### Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('greenlang.security.kms')
logger.setLevel(logging.DEBUG)
```

## Migration from Other Signing Methods

### From Ephemeral Keys

```python
# Before (ephemeral)
os.environ['GL_SIGNING_MODE'] = 'ephemeral'
signer = create_signer()

# After (KMS)
os.environ['GL_SIGNING_MODE'] = 'kms'
os.environ['GL_KMS_KEY_ID'] = 'your-kms-key'
signer = create_signer()  # Same API!
```

### From Sigstore

```python
# Both can coexist
if os.environ.get('CI'):
    # Use Sigstore in CI/CD
    config = SigningConfig(mode='keyless')
else:
    # Use KMS in production
    config = SigningConfig(mode='kms')

signer = create_signer(config)
```

## Cost Optimization

1. **Enable Caching**: Reduce API calls with appropriate cache TTL
2. **Batch Operations**: Use batch signing for multiple documents
3. **Regional Keys**: Use regional keys instead of global when possible
4. **Key Lifecycle**: Delete unused keys and versions
5. **Monitor Usage**: Track API calls and optimize patterns

## Compliance

The KMS integration supports various compliance requirements:

- **FIPS 140-2**: All providers support FIPS-validated cryptographic modules
- **PCI DSS**: Key management aligned with PCI DSS requirements
- **SOC 2**: Audit trails and access controls
- **GDPR**: Data residency controls with regional keys
- **HIPAA**: Encryption key management for healthcare data