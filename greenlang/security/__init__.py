"""
GreenLang Security Module Facade
=================================

Re-exports security utilities from core.greenlang.security for easier importing.
This facade ensures backward compatibility and provides a cleaner import path.
"""

import sys
import importlib.util
from pathlib import Path

# Get paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
core_security_dir = project_root / "core" / "greenlang" / "security"

# Function to load module from file
def load_module_from_file(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load core security modules
network = load_module_from_file("_core_security_network", core_security_dir / "network.py")
paths = load_module_from_file("_core_security_paths", core_security_dir / "paths.py")
signatures = load_module_from_file("_core_security_signatures", core_security_dir / "signatures.py")

# Import from network module
from _core_security_network import (
    create_secure_session,
    validate_url,
    validate_git_url,
    safe_download,
    SecureHTTPAdapter,
    create_secure_ssl_context,
)

# Import from paths module
from _core_security_paths import (
    validate_safe_path,
    safe_extract_tar,
    safe_extract_zip,
    safe_extract_archive,
    validate_pack_structure,
    safe_create_directory,
)

# Import from signatures module
from _core_security_signatures import (
    PackVerifier,
    SignatureVerificationError,
    verify_pack_integrity
)

# Also export from local modules
from .key_manager import KeyManager, KeyType, KeyPurpose, KeyMetadata, get_key_manager
from .signing import (
    SignResult,
    SigningConfig,
    Signer,
    Verifier,
    SigstoreKeylessSigner,
    EphemeralKeypairSigner,
    ExternalKMSSigner,
    SigstoreBundleVerifier,
    DetachedSigVerifier,
    create_signer,
    create_verifier,
    sign_artifact,
    verify_artifact
)

__all__ = [
    # Network security
    'create_secure_session',
    'validate_url',
    'validate_git_url',
    'safe_download',
    'SecureHTTPAdapter',
    'create_secure_ssl_context',
    # Path security
    'validate_safe_path',
    'safe_extract_tar',
    'safe_extract_zip',
    'safe_extract_archive',
    'validate_pack_structure',
    'safe_create_directory',
    # Signature verification
    'PackVerifier',
    'SignatureVerificationError',
    'verify_pack_integrity',
    # Key management
    'KeyManager',
    'KeyType',
    'KeyPurpose',
    'KeyMetadata',
    'get_key_manager',
    # Signing
    'SignResult',
    'SigningConfig',
    'Signer',
    'Verifier',
    'SigstoreKeylessSigner',
    'EphemeralKeypairSigner',
    'ExternalKMSSigner',
    'SigstoreBundleVerifier',
    'DetachedSigVerifier',
    'create_signer',
    'create_verifier',
    'sign_artifact',
    'verify_artifact'
]