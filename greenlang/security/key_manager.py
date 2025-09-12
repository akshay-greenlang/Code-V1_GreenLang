"""
Production Key Management System for GreenLang

This module provides secure key management for pack signing,
verification, and other cryptographic operations.
"""

import os
import json
import base64
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
    from cryptography.hazmat.backends import default_backend
    from cryptography.x509 import (
        CertificateBuilder, Name, NameAttribute,
        BasicConstraints, KeyUsage, ExtendedKeyUsage,
        ExtensionType, SubjectAlternativeName, DNSName
    )
    from cryptography.x509.oid import NameOID, ExtensionOID, ExtendedKeyUsageOID
    import cryptography.x509 as x509
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class KeyType(Enum):
    """Types of cryptographic keys"""
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"
    ECDSA_P256 = "ecdsa-p256"
    ECDSA_P384 = "ecdsa-p384"
    ED25519 = "ed25519"


class KeyPurpose(Enum):
    """Purpose of cryptographic keys"""
    SIGNING = "signing"
    ENCRYPTION = "encryption"
    KEY_AGREEMENT = "key-agreement"
    AUTHENTICATION = "authentication"


@dataclass
class KeyMetadata:
    """Metadata for a cryptographic key"""
    key_id: str
    key_type: KeyType
    purpose: KeyPurpose
    created_at: datetime
    expires_at: Optional[datetime] = None
    owner: Optional[str] = None
    description: Optional[str] = None
    algorithm: Optional[str] = None
    public_key_hash: Optional[str] = None
    certificate_serial: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    rotation_policy: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        data = asdict(self)
        data['key_type'] = self.key_type.value
        data['purpose'] = self.purpose.value
        data['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeyMetadata':
        """Create metadata from dictionary"""
        data['key_type'] = KeyType(data['key_type'])
        data['purpose'] = KeyPurpose(data['purpose'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if key has expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at


class KeyManager:
    """
    Production-ready key management system for GreenLang.
    
    Features:
    - Secure key generation and storage
    - Key rotation and expiration
    - Certificate management
    - Hardware security module (HSM) support (future)
    - Key distribution and discovery
    """
    
    def __init__(self, key_store_dir: Optional[Path] = None):
        """
        Initialize the key manager.
        
        Args:
            key_store_dir: Directory for storing keys (default: ~/.greenlang/keys)
        """
        self.key_store_dir = key_store_dir or Path.home() / ".greenlang" / "keys"
        self.key_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for organization
        self.private_keys_dir = self.key_store_dir / "private"
        self.public_keys_dir = self.key_store_dir / "public"
        self.certificates_dir = self.key_store_dir / "certificates"
        self.metadata_dir = self.key_store_dir / "metadata"
        self.revoked_dir = self.key_store_dir / "revoked"
        
        for dir in [self.private_keys_dir, self.public_keys_dir, 
                   self.certificates_dir, self.metadata_dir, self.revoked_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Set secure permissions on private key directory (Unix-like systems)
        if os.name != 'nt':
            os.chmod(self.private_keys_dir, 0o700)
        
        # Load key registry
        self.keys: Dict[str, KeyMetadata] = {}
        self._load_key_registry()
    
    def _load_key_registry(self) -> None:
        """Load existing key metadata from disk"""
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                metadata = KeyMetadata.from_dict(data)
                self.keys[metadata.key_id] = metadata
            except Exception as e:
                print(f"Warning: Could not load key metadata from {metadata_file}: {e}")
    
    def _save_key_metadata(self, metadata: KeyMetadata) -> None:
        """Save key metadata to disk"""
        metadata_file = self.metadata_dir / f"{metadata.key_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def generate_key(self,
                    key_id: str,
                    key_type: KeyType = KeyType.RSA_2048,
                    purpose: KeyPurpose = KeyPurpose.SIGNING,
                    owner: Optional[str] = None,
                    expires_in_days: int = 365,
                    passphrase: Optional[bytes] = None) -> Tuple[str, str]:
        """
        Generate a new cryptographic key pair.
        
        Args:
            key_id: Unique identifier for the key
            key_type: Type of key to generate
            purpose: Purpose of the key
            owner: Owner of the key
            expires_in_days: Key expiration in days
            passphrase: Optional passphrase for private key encryption
        
        Returns:
            Tuple of (private_key_path, public_key_path)
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        # Check if key already exists
        if key_id in self.keys:
            raise ValueError(f"Key with ID '{key_id}' already exists")
        
        # Generate key based on type
        if key_type in [KeyType.RSA_2048, KeyType.RSA_4096]:
            key_size = 2048 if key_type == KeyType.RSA_2048 else 4096
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            algorithm = f"RSA-{key_size}"
        elif key_type == KeyType.ECDSA_P256:
            private_key = ec.generate_private_key(
                ec.SECP256R1(),
                backend=default_backend()
            )
            algorithm = "ECDSA-P256"
        elif key_type == KeyType.ECDSA_P384:
            private_key = ec.generate_private_key(
                ec.SECP384R1(),
                backend=default_backend()
            )
            algorithm = "ECDSA-P384"
        else:
            raise ValueError(f"Unsupported key type: {key_type}")
        
        # Serialize private key
        encryption = serialization.NoEncryption()
        if passphrase:
            encryption = serialization.BestAvailableEncryption(passphrase)
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption
        )
        
        # Serialize public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Calculate public key hash
        public_key_hash = hashlib.sha256(public_pem).hexdigest()
        
        # Save keys to disk
        private_key_file = self.private_keys_dir / f"{key_id}.pem"
        public_key_file = self.public_keys_dir / f"{key_id}.pub"
        
        with open(private_key_file, 'wb') as f:
            f.write(private_pem)
        
        with open(public_key_file, 'wb') as f:
            f.write(public_pem)
        
        # Set secure permissions on private key
        if os.name != 'nt':
            os.chmod(private_key_file, 0o600)
        
        # Create metadata
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            purpose=purpose,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days),
            owner=owner,
            algorithm=algorithm,
            public_key_hash=public_key_hash
        )
        
        # Save metadata
        self.keys[key_id] = metadata
        self._save_key_metadata(metadata)
        
        return str(private_key_file), str(public_key_file)
    
    def generate_certificate(self,
                           key_id: str,
                           subject: Dict[str, str],
                           issuer: Optional[Dict[str, str]] = None,
                           validity_days: int = 365,
                           is_ca: bool = False,
                           san_dns: Optional[List[str]] = None) -> str:
        """
        Generate a X.509 certificate for a key.
        
        Args:
            key_id: ID of the key to generate certificate for
            subject: Subject information (CN, O, OU, etc.)
            issuer: Issuer information (None for self-signed)
            validity_days: Certificate validity in days
            is_ca: Whether this is a CA certificate
            san_dns: Subject Alternative Names (DNS)
        
        Returns:
            Path to the generated certificate
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        if key_id not in self.keys:
            raise ValueError(f"Key '{key_id}' not found")
        
        # Load private key
        private_key_file = self.private_keys_dir / f"{key_id}.pem"
        with open(private_key_file, 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )
        
        # Build subject
        subject_components = []
        if 'CN' in subject:
            subject_components.append(NameAttribute(NameOID.COMMON_NAME, subject['CN']))
        if 'O' in subject:
            subject_components.append(NameAttribute(NameOID.ORGANIZATION_NAME, subject['O']))
        if 'OU' in subject:
            subject_components.append(NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, subject['OU']))
        if 'C' in subject:
            subject_components.append(NameAttribute(NameOID.COUNTRY_NAME, subject['C']))
        
        subject_name = Name(subject_components)
        
        # Issuer (self-signed if not provided)
        issuer_name = subject_name if not issuer else Name([
            NameAttribute(getattr(NameOID, k), v) for k, v in issuer.items()
        ])
        
        # Create certificate builder
        builder = CertificateBuilder()
        builder = builder.subject_name(subject_name)
        builder = builder.issuer_name(issuer_name)
        builder = builder.public_key(private_key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.utcnow())
        builder = builder.not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        )
        
        # Add extensions
        builder = builder.add_extension(
            BasicConstraints(ca=is_ca, path_length=None if not is_ca else 0),
            critical=True
        )
        
        builder = builder.add_extension(
            KeyUsage(
                digital_signature=True,
                key_encipherment=not is_ca,
                content_commitment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=is_ca,
                crl_sign=is_ca,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True
        )
        
        if not is_ca:
            builder = builder.add_extension(
                ExtendedKeyUsage([
                    ExtendedKeyUsageOID.CODE_SIGNING,
                    ExtendedKeyUsageOID.CLIENT_AUTH
                ]),
                critical=True
            )
        
        # Add SAN if provided
        if san_dns:
            builder = builder.add_extension(
                SubjectAlternativeName([DNSName(dns) for dns in san_dns]),
                critical=False
            )
        
        # Sign certificate
        certificate = builder.sign(private_key, hashes.SHA256(), backend=default_backend())
        
        # Save certificate
        cert_file = self.certificates_dir / f"{key_id}.crt"
        with open(cert_file, 'wb') as f:
            f.write(certificate.public_bytes(serialization.Encoding.PEM))
        
        # Update metadata
        metadata = self.keys[key_id]
        metadata.certificate_serial = str(certificate.serial_number)
        self._save_key_metadata(metadata)
        
        return str(cert_file)
    
    def import_key(self,
                  key_id: str,
                  private_key_path: Optional[Path] = None,
                  public_key_path: Optional[Path] = None,
                  certificate_path: Optional[Path] = None,
                  purpose: KeyPurpose = KeyPurpose.SIGNING,
                  owner: Optional[str] = None) -> None:
        """Import existing keys into the key manager"""
        if key_id in self.keys:
            raise ValueError(f"Key '{key_id}' already exists")
        
        # Copy files to key store
        if private_key_path and private_key_path.exists():
            import shutil
            dest = self.private_keys_dir / f"{key_id}.pem"
            shutil.copy2(private_key_path, dest)
            if os.name != 'nt':
                os.chmod(dest, 0o600)
        
        if public_key_path and public_key_path.exists():
            import shutil
            dest = self.public_keys_dir / f"{key_id}.pub"
            shutil.copy2(public_key_path, dest)
        
        if certificate_path and certificate_path.exists():
            import shutil
            dest = self.certificates_dir / f"{key_id}.crt"
            shutil.copy2(certificate_path, dest)
        
        # Create metadata
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=KeyType.RSA_2048,  # Default, should detect from key
            purpose=purpose,
            created_at=datetime.utcnow(),
            owner=owner
        )
        
        self.keys[key_id] = metadata
        self._save_key_metadata(metadata)
    
    def get_key_path(self, key_id: str, key_type: str = "private") -> Optional[Path]:
        """Get path to a key file"""
        if key_id not in self.keys:
            return None
        
        if key_type == "private":
            path = self.private_keys_dir / f"{key_id}.pem"
        elif key_type == "public":
            path = self.public_keys_dir / f"{key_id}.pub"
        elif key_type == "certificate":
            path = self.certificates_dir / f"{key_id}.crt"
        else:
            return None
        
        return path if path.exists() else None
    
    def list_keys(self,
                 purpose: Optional[KeyPurpose] = None,
                 owner: Optional[str] = None,
                 include_expired: bool = False) -> List[KeyMetadata]:
        """List keys with optional filtering"""
        keys = list(self.keys.values())
        
        if purpose:
            keys = [k for k in keys if k.purpose == purpose]
        
        if owner:
            keys = [k for k in keys if k.owner == owner]
        
        if not include_expired:
            keys = [k for k in keys if not k.is_expired()]
        
        return keys
    
    def rotate_key(self, key_id: str, keep_old: bool = True) -> Tuple[str, str]:
        """Rotate a key by generating a new one and optionally archiving the old"""
        if key_id not in self.keys:
            raise ValueError(f"Key '{key_id}' not found")
        
        old_metadata = self.keys[key_id]
        
        # Archive old key if requested
        if keep_old:
            import shutil
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            archive_id = f"{key_id}_{timestamp}"
            
            # Move old files to archive
            for ext, subdir in [(".pem", self.private_keys_dir),
                               (".pub", self.public_keys_dir),
                               (".crt", self.certificates_dir)]:
                old_file = subdir / f"{key_id}{ext}"
                if old_file.exists():
                    archive_file = subdir / f"{archive_id}{ext}"
                    shutil.move(old_file, archive_file)
            
            # Update old metadata
            old_metadata.key_id = archive_id
            old_metadata.tags['rotated_from'] = key_id
            old_metadata.tags['rotated_at'] = timestamp
            self.keys[archive_id] = old_metadata
            self._save_key_metadata(old_metadata)
        
        # Generate new key with same parameters
        return self.generate_key(
            key_id=key_id,
            key_type=old_metadata.key_type,
            purpose=old_metadata.purpose,
            owner=old_metadata.owner,
            expires_in_days=365
        )
    
    def revoke_key(self, key_id: str, reason: str = "unspecified") -> None:
        """Revoke a key and move it to revoked directory"""
        if key_id not in self.keys:
            raise ValueError(f"Key '{key_id}' not found")
        
        import shutil
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Move files to revoked directory
        for ext, subdir in [(".pem", self.private_keys_dir),
                           (".pub", self.public_keys_dir),
                           (".crt", self.certificates_dir)]:
            src = subdir / f"{key_id}{ext}"
            if src.exists():
                dst = self.revoked_dir / f"{key_id}_{timestamp}{ext}"
                shutil.move(src, dst)
        
        # Update metadata
        metadata = self.keys[key_id]
        metadata.tags['revoked_at'] = timestamp
        metadata.tags['revocation_reason'] = reason
        
        # Save revocation record
        revocation_file = self.revoked_dir / f"{key_id}_{timestamp}.json"
        with open(revocation_file, 'w') as f:
            json.dump({
                'key_id': key_id,
                'revoked_at': timestamp,
                'reason': reason,
                'metadata': metadata.to_dict()
            }, f, indent=2)
        
        # Remove from active keys
        del self.keys[key_id]
        metadata_file = self.metadata_dir / f"{key_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
    
    def export_public_keys(self, output_dir: Path) -> int:
        """Export all public keys to a directory for distribution"""
        output_dir.mkdir(parents=True, exist_ok=True)
        exported = 0
        
        for key_id, metadata in self.keys.items():
            if metadata.is_expired():
                continue
            
            public_key_file = self.public_keys_dir / f"{key_id}.pub"
            if public_key_file.exists():
                import shutil
                shutil.copy2(public_key_file, output_dir / f"{key_id}.pub")
                
                # Also export metadata
                meta_file = output_dir / f"{key_id}.json"
                with open(meta_file, 'w') as f:
                    json.dump({
                        'key_id': key_id,
                        'owner': metadata.owner,
                        'purpose': metadata.purpose.value,
                        'algorithm': metadata.algorithm,
                        'public_key_hash': metadata.public_key_hash,
                        'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None
                    }, f, indent=2)
                
                exported += 1
        
        return exported
    
    def verify_key_integrity(self, key_id: str) -> bool:
        """Verify integrity of stored keys"""
        if key_id not in self.keys:
            return False
        
        metadata = self.keys[key_id]
        
        # Check if public key hash matches
        public_key_file = self.public_keys_dir / f"{key_id}.pub"
        if public_key_file.exists():
            with open(public_key_file, 'rb') as f:
                public_key_data = f.read()
            
            calculated_hash = hashlib.sha256(public_key_data).hexdigest()
            if metadata.public_key_hash and calculated_hash != metadata.public_key_hash:
                return False
        
        # Check expiration
        if metadata.is_expired():
            return False
        
        return True


# Global key manager instance
_key_manager: Optional[KeyManager] = None


def get_key_manager(key_store_dir: Optional[Path] = None) -> KeyManager:
    """Get or create the global key manager instance"""
    global _key_manager
    if _key_manager is None:
        _key_manager = KeyManager(key_store_dir)
    return _key_manager