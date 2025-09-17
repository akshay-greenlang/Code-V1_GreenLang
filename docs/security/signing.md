# GreenLang Secure Signing Architecture

## Overview

GreenLang implements a secure, provider-based signing architecture that **eliminates all hardcoded keys** from the codebase. This document describes the signing mechanisms, security guarantees, and usage patterns.

## Key Security Principles

### ✅ NO HARDCODED KEYS
- **Zero embedded private keys** in source code
- **Zero mock keys** in production paths
- **Zero default passwords** or passphrases
- All keys are either:
  - Ephemeral (generated at runtime for tests)
  - External (Sigstore keyless via OIDC)
  - KMS-managed (future implementation)

### ✅ Default-Deny Verification
- Unsigned artifacts are **rejected by default**
- Explicit `--allow-unsigned` flag required with warning
- Policy enforcement at install time

### ✅ Cryptographic Agility
- Provider abstraction supports multiple algorithms
- Easy migration between signing methods
- Future-proof for quantum-resistant algorithms

## Signing Providers

### 1. Sigstore Keyless (Production)

Used in CI/CD environments with OIDC identity.

**How it works:**
1. GitHub Actions provides OIDC token
2. Fulcio issues short-lived certificate
3. Artifact is signed with ephemeral key
4. Signature logged to Rekor transparency log

**Configuration:**
```bash
# Automatically enabled in CI
export GL_SIGNING_MODE=keyless
```

**Usage:**
```python
from greenlang.security.signing import create_signer

signer = create_signer()  # Auto-detects CI environment
result = signer.sign(payload)
```

### 2. Ephemeral Keypair (Testing)

Generates new Ed25519 keypair for each test run.

**Characteristics:**
- Keys exist only in memory
- New keys for each test instance
- No key persistence between runs

**Configuration:**
```bash
# Default for local development
export GL_SIGNING_MODE=ephemeral
```

### 3. External KMS (Future)

Placeholder for enterprise KMS integration:
- AWS KMS
- HashiCorp Vault
- Azure Key Vault
- Google Cloud KMS

## CLI Usage

### Publishing with Signatures

```bash
# Sign during publish (default)
gl pack publish

# Publish without signing (not recommended)
GL_ALLOW_UNSIGNED=1 gl pack publish --no-sign
```

### Installing with Verification

```bash
# Verify signatures (default)
gl pack install mypack

# Install unsigned pack (dangerous!)
gl pack install mypack --allow-unsigned
```

### Manual Verification

```bash
# Verify pack signature
gl verify pack.yaml --sig pack.sig

# Verify with Sigstore bundle
gl verify artifact.tar.gz --bundle artifact.bundle
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GL_SIGNING_MODE` | Signing provider mode | `keyless` in CI, `ephemeral` locally |
| `GL_SIGSTORE_STAGING` | Use Sigstore staging | `0` (production) |
| `GL_ALLOW_UNSIGNED` | Allow unsigned artifacts | `0` (reject) |
| `GL_KMS_KEY_ID` | KMS key identifier | None |

## Security Workflows

### Release Signing (CI/CD)

```yaml
# .github/workflows/release.yml
- name: Sign with Sigstore
  env:
    GL_SIGNING_MODE: keyless
  run: |
    gl pack publish --sign
```

### Local Development

```bash
# Tests use ephemeral keys automatically
pytest tests/

# Local pack development
GL_SIGNING_MODE=ephemeral gl pack publish --registry local
```

## Verification Process

### Standard Verification Flow

```python
from greenlang.security.signing import verify_artifact

# Verifier auto-detects signature type
is_valid = verify_artifact(
    artifact_path=Path("pack.tar.gz"),
    signature=signature_dict
)
```

### Manual Verification

```bash
# Verify Sigstore signature
python -m sigstore verify identity artifact.tar.gz \
  --cert-oidc-issuer https://token.actions.githubusercontent.com \
  --cert-identity https://github.com/org/repo/.github/workflows/release.yml@refs/tags/v1.0.0
```

## Migration from Legacy

### Detecting Legacy Mock Signatures

Legacy mock signatures are detected and rejected:

```python
if signature["spec"]["signature"]["algorithm"] == "mock":
    raise SecurityError("Mock signatures are no longer supported")
```

### Migration Path

1. **Update CI/CD** to use `GL_SIGNING_MODE=keyless`
2. **Update tests** to use ephemeral fixtures
3. **Remove any** `.key`, `.pem` files from repository
4. **Enable secret scanning** in CI to prevent regression

## Security Guarantees

### What We Protect Against

- ✅ **Supply chain attacks**: All artifacts cryptographically signed
- ✅ **Key compromise**: Keyless signing eliminates long-lived keys
- ✅ **Insider threats**: Transparency logs provide audit trail
- ✅ **Tampering**: Hash verification before signature check

### What We Don't Protect Against

- ❌ Compromised CI/CD environment (mitigated by OIDC)
- ❌ Malicious pack content (use policy enforcement)
- ❌ Zero-day in signing libraries (keep dependencies updated)

## Troubleshooting

### Common Issues

**"No signature found"**
- Ensure pack was published with `--sign` flag
- Check `GL_SIGNING_MODE` is not `disabled`

**"Signature verification failed"**
- Verify artifact hasn't been modified
- Check certificate hasn't expired (Sigstore certs valid 10 min)
- Ensure correct verifier for signature type

**"Mock signatures deprecated"**
- Update to new signing architecture
- Regenerate signatures with current provider

### Debug Mode

```bash
# Enable verbose logging
export GL_LOG_LEVEL=DEBUG

# Test signing provider
python -c "from greenlang.security.signing import create_signer; print(create_signer().get_signer_info())"
```

## Compliance

### Standards Compliance

- **SLSA Level 3**: Signed provenance, non-falsifiable
- **NIST 800-147**: Secure boot and attestation
- **PCI DSS 10.2**: Audit trails via transparency logs

### Audit Requirements

All signed artifacts include:
- Timestamp
- Signer identity
- Hash algorithm and value
- Transparency log entry (Sigstore)

## Best Practices

1. **Never disable verification** in production
2. **Use keyless signing** in CI/CD
3. **Rotate KMS keys** regularly (when implemented)
4. **Monitor transparency logs** for unauthorized signatures
5. **Scan for secrets** in every PR

## API Reference

### Signer Interface

```python
class Signer(ABC):
    @abstractmethod
    def sign(self, payload: bytes) -> SignResult:
        """Sign a payload"""
        pass

    @abstractmethod
    def get_signer_info(self) -> Dict[str, Any]:
        """Get signer information"""
        pass
```

### Verifier Interface

```python
class Verifier(ABC):
    @abstractmethod
    def verify(self, payload: bytes, signature: bytes, **kwargs) -> None:
        """Verify a signature (raises InvalidSignature on failure)"""
        pass
```

### Configuration

```python
@dataclass
class SigningConfig:
    mode: str  # 'keyless', 'ephemeral', 'kms', 'disabled'
    kms_key_id: Optional[str] = None
    sigstore_audience: Optional[str] = None
    sigstore_staging: bool = False
```

## Security Contact

Report security issues to: security@greenlang.org

Do NOT create public issues for security vulnerabilities.