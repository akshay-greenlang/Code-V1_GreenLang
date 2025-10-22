"""
Test suite for encryption utilities.

Tests encryption/decryption of sensitive ESG data, ensuring compliance
with data protection regulations (GDPR, CSRD).
"""

import pytest
import os
import base64
from cryptography.fernet import Fernet

# Import the encryption module
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.encryption import EncryptionManager, get_encryption_manager


class TestEncryptionManager:
    """Test cases for EncryptionManager class."""

    @pytest.fixture
    def encryption_key(self):
        """Generate a test encryption key."""
        return Fernet.generate_key()

    @pytest.fixture
    def encryption_manager(self, encryption_key):
        """Create an EncryptionManager instance with test key."""
        return EncryptionManager(key=encryption_key)

    def test_initialization_with_key(self, encryption_key):
        """Test EncryptionManager initialization with provided key."""
        em = EncryptionManager(key=encryption_key)
        assert em is not None
        assert em.fernet is not None

    def test_initialization_without_key_raises_error(self, monkeypatch):
        """Test that initialization without key and env variable raises error."""
        # Remove environment variable if it exists
        monkeypatch.delenv('CSRD_ENCRYPTION_KEY', raising=False)

        with pytest.raises(ValueError) as exc_info:
            EncryptionManager()

        assert "CSRD_ENCRYPTION_KEY not set" in str(exc_info.value)

    def test_initialization_from_env(self, monkeypatch, encryption_key):
        """Test EncryptionManager initialization from environment variable."""
        # Set environment variable
        key_b64 = base64.urlsafe_b64encode(encryption_key).decode()
        monkeypatch.setenv('CSRD_ENCRYPTION_KEY', key_b64)

        em = EncryptionManager()
        assert em is not None
        assert em.fernet is not None

    def test_generate_key(self):
        """Test key generation."""
        key = EncryptionManager.generate_key()
        assert key is not None
        assert isinstance(key, bytes)
        assert len(key) == 44  # Fernet keys are 44 bytes when base64 encoded

    def test_encrypt_string(self, encryption_manager):
        """Test encrypting a string."""
        original_data = "sensitive ESG data 12345"
        encrypted = encryption_manager.encrypt(original_data)

        assert encrypted is not None
        assert isinstance(encrypted, str)
        assert encrypted != original_data
        assert len(encrypted) > 0

    def test_encrypt_bytes(self, encryption_manager):
        """Test encrypting bytes."""
        original_data = b"sensitive bytes data"
        encrypted = encryption_manager.encrypt(original_data)

        assert encrypted is not None
        assert isinstance(encrypted, str)
        assert len(encrypted) > 0

    def test_decrypt_string(self, encryption_manager):
        """Test decrypting data back to original string."""
        original_data = "sensitive data 98765"
        encrypted = encryption_manager.encrypt(original_data)
        decrypted = encryption_manager.decrypt(encrypted)

        assert decrypted == original_data

    def test_encrypt_decrypt_roundtrip(self, encryption_manager):
        """Test full encryption-decryption roundtrip."""
        test_cases = [
            "Simple text",
            "Text with numbers 123456",
            "Special chars: !@#$%^&*()",
            "Unicode: €ñ中文",
            "Multi\nline\ntext",
            "Very long text " * 100,
            "",  # Empty string
        ]

        for original in test_cases:
            encrypted = encryption_manager.encrypt(original)
            decrypted = encryption_manager.decrypt(encrypted)
            assert decrypted == original, f"Failed for: {original}"

    def test_encrypt_dict_single_field(self, encryption_manager):
        """Test encrypting a single field in a dictionary."""
        data = {
            "company_name": "Acme Corp",
            "revenue_eur": 45000000,
            "country": "Germany"
        }

        encrypted_data = encryption_manager.encrypt_dict(data, ['revenue_eur'])

        # Check non-encrypted fields remain unchanged
        assert encrypted_data['company_name'] == "Acme Corp"
        assert encrypted_data['country'] == "Germany"

        # Check encrypted field is different
        assert encrypted_data['revenue_eur'] != 45000000
        assert isinstance(encrypted_data['revenue_eur'], str)

        # Check encryption flag is set
        assert encrypted_data['revenue_eur_encrypted'] is True

    def test_encrypt_dict_multiple_fields(self, encryption_manager):
        """Test encrypting multiple fields in a dictionary."""
        data = {
            "company_name": "Acme Corp",
            "revenue_eur": 45000000,
            "lei_code": "5493001234567890",
            "tax_id": "DE123456789",
            "country": "Germany"
        }

        fields_to_encrypt = ['revenue_eur', 'lei_code', 'tax_id']
        encrypted_data = encryption_manager.encrypt_dict(data, fields_to_encrypt)

        # Check non-encrypted fields
        assert encrypted_data['company_name'] == "Acme Corp"
        assert encrypted_data['country'] == "Germany"

        # Check encrypted fields are different
        assert encrypted_data['revenue_eur'] != 45000000
        assert encrypted_data['lei_code'] != "5493001234567890"
        assert encrypted_data['tax_id'] != "DE123456789"

        # Check encryption flags
        assert encrypted_data['revenue_eur_encrypted'] is True
        assert encrypted_data['lei_code_encrypted'] is True
        assert encrypted_data['tax_id_encrypted'] is True

    def test_encrypt_dict_with_none_values(self, encryption_manager):
        """Test that None values are not encrypted."""
        data = {
            "company_name": "Acme Corp",
            "revenue_eur": None,
            "lei_code": "5493001234567890"
        }

        encrypted_data = encryption_manager.encrypt_dict(data, ['revenue_eur', 'lei_code'])

        # None should remain None
        assert encrypted_data['revenue_eur'] is None
        assert 'revenue_eur_encrypted' not in encrypted_data

        # Non-None field should be encrypted
        assert encrypted_data['lei_code'] != "5493001234567890"
        assert encrypted_data['lei_code_encrypted'] is True

    def test_encrypt_dict_with_missing_fields(self, encryption_manager):
        """Test encrypting fields that don't exist in the dictionary."""
        data = {
            "company_name": "Acme Corp",
            "revenue_eur": 45000000
        }

        # Try to encrypt a field that doesn't exist
        encrypted_data = encryption_manager.encrypt_dict(data, ['revenue_eur', 'non_existent_field'])

        # Should only encrypt existing field
        assert encrypted_data['revenue_eur'] != 45000000
        assert encrypted_data['revenue_eur_encrypted'] is True
        assert 'non_existent_field' not in encrypted_data

    def test_decrypt_dict_single_field(self, encryption_manager):
        """Test decrypting a single field in a dictionary."""
        data = {
            "company_name": "Acme Corp",
            "revenue_eur": 45000000,
        }

        # Encrypt then decrypt
        encrypted_data = encryption_manager.encrypt_dict(data, ['revenue_eur'])
        decrypted_data = encryption_manager.decrypt_dict(encrypted_data, ['revenue_eur'])

        # Check decrypted value (note: will be string representation)
        assert decrypted_data['revenue_eur'] == str(45000000)
        assert decrypted_data['revenue_eur_encrypted'] is False

    def test_decrypt_dict_multiple_fields(self, encryption_manager):
        """Test decrypting multiple fields in a dictionary."""
        data = {
            "company_name": "Acme Corp",
            "revenue_eur": 45000000,
            "lei_code": "5493001234567890",
            "country": "Germany"
        }

        # Encrypt then decrypt
        encrypted_data = encryption_manager.encrypt_dict(data, ['revenue_eur', 'lei_code'])
        decrypted_data = encryption_manager.decrypt_dict(encrypted_data, ['revenue_eur', 'lei_code'])

        # Check decrypted values
        assert decrypted_data['revenue_eur'] == str(45000000)
        assert decrypted_data['lei_code'] == "5493001234567890"
        assert decrypted_data['revenue_eur_encrypted'] is False
        assert decrypted_data['lei_code_encrypted'] is False

        # Non-encrypted field should remain unchanged
        assert decrypted_data['company_name'] == "Acme Corp"

    def test_decrypt_dict_without_encryption_flag(self, encryption_manager):
        """Test decrypting fields without encryption flags doesn't fail."""
        data = {
            "company_name": "Acme Corp",
            "revenue_eur": "plain text value"
        }

        # Try to decrypt without encryption flag
        decrypted_data = encryption_manager.decrypt_dict(data, ['revenue_eur'])

        # Should return data unchanged if no encryption flag
        assert decrypted_data['revenue_eur'] == "plain text value"

    def test_full_roundtrip_with_dict(self, encryption_manager):
        """Test complete encrypt-decrypt roundtrip with dictionary."""
        original_data = {
            "company_name": "Green Manufacturing Ltd",
            "revenue_eur": 125000000,
            "lei_code": "5493001234567890ABCD",
            "tax_id": "GB987654321",
            "country": "United Kingdom",
            "employee_count": 450
        }

        sensitive_fields = ['revenue_eur', 'lei_code', 'tax_id']

        # Encrypt
        encrypted_data = encryption_manager.encrypt_dict(original_data, sensitive_fields)

        # Verify encryption
        for field in sensitive_fields:
            assert encrypted_data[field] != str(original_data[field])
            assert encrypted_data[f'{field}_encrypted'] is True

        # Decrypt
        decrypted_data = encryption_manager.decrypt_dict(encrypted_data, sensitive_fields)

        # Verify decryption
        assert decrypted_data['revenue_eur'] == str(original_data['revenue_eur'])
        assert decrypted_data['lei_code'] == original_data['lei_code']
        assert decrypted_data['tax_id'] == original_data['tax_id']

        # Non-encrypted fields should be unchanged
        assert decrypted_data['company_name'] == original_data['company_name']
        assert decrypted_data['country'] == original_data['country']

    def test_different_keys_produce_different_ciphertext(self):
        """Test that different keys produce different encrypted output."""
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()

        em1 = EncryptionManager(key=key1)
        em2 = EncryptionManager(key=key2)

        data = "sensitive data"

        encrypted1 = em1.encrypt(data)
        encrypted2 = em2.encrypt(data)

        assert encrypted1 != encrypted2

    def test_wrong_key_cannot_decrypt(self):
        """Test that data encrypted with one key cannot be decrypted with another."""
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()

        em1 = EncryptionManager(key=key1)
        em2 = EncryptionManager(key=key2)

        data = "sensitive data"
        encrypted = em1.encrypt(data)

        # Attempting to decrypt with wrong key should raise an error
        with pytest.raises(Exception):  # Fernet raises InvalidToken
            em2.decrypt(encrypted)


class TestGetEncryptionManager:
    """Test cases for get_encryption_manager singleton function."""

    def test_singleton_returns_same_instance(self, monkeypatch):
        """Test that get_encryption_manager returns the same instance."""
        # Set environment variable
        key = Fernet.generate_key()
        key_b64 = base64.urlsafe_b64encode(key).decode()
        monkeypatch.setenv('CSRD_ENCRYPTION_KEY', key_b64)

        # Reset singleton
        import utils.encryption
        utils.encryption._encryption_manager = None

        em1 = get_encryption_manager()
        em2 = get_encryption_manager()

        assert em1 is em2


class TestEncryptionIntegration:
    """Integration tests simulating real-world usage."""

    @pytest.fixture
    def test_key(self):
        """Generate a test key."""
        return Fernet.generate_key()

    def test_esg_report_encryption(self, test_key):
        """Test encrypting a complete ESG report data structure."""
        em = EncryptionManager(key=test_key)

        esg_report = {
            "company_name": "Sustainable Tech GmbH",
            "reporting_year": 2024,
            "country": "DE",
            "revenue_eur": 250000000,
            "lei_code": "549300ABCDEF1234567890",
            "tax_id": "DE987654321",
            "employees": 1200,
            "ghg_emissions_scope1": 5000,
            "ghg_emissions_scope2": 8000,
            "water_consumption": 150000,
            "employee_salaries": {"average": 65000, "median": 60000},
            "executive_compensation": [250000, 180000, 175000]
        }

        sensitive_fields = [
            'revenue_eur',
            'lei_code',
            'tax_id',
            'employee_salaries',
            'executive_compensation'
        ]

        # Encrypt
        encrypted_report = em.encrypt_dict(esg_report, sensitive_fields)

        # Verify non-sensitive data is readable
        assert encrypted_report['company_name'] == "Sustainable Tech GmbH"
        assert encrypted_report['employees'] == 1200

        # Verify sensitive data is encrypted
        assert encrypted_report['revenue_eur'] != 250000000

        # Decrypt
        decrypted_report = em.decrypt_dict(encrypted_report, sensitive_fields)

        # Verify all data is restored correctly
        assert decrypted_report['revenue_eur'] == str(250000000)
        assert decrypted_report['lei_code'] == "549300ABCDEF1234567890"

    def test_materiality_assessment_encryption(self, test_key):
        """Test encrypting materiality assessment data."""
        em = EncryptionManager(key=test_key)

        materiality_data = {
            "topic": "Climate Change",
            "financial_materiality_score": 8.5,
            "impact_materiality_score": 9.0,
            "stakeholder_feedback": "Confidential stakeholder concerns about transition risks",
            "confidential_impact_data": "Internal analysis of stranded assets",
            "public_disclosure": "Company acknowledges climate risks"
        }

        sensitive_fields = ['stakeholder_feedback', 'confidential_impact_data']

        encrypted_data = em.encrypt_dict(materiality_data, sensitive_fields)
        decrypted_data = em.decrypt_dict(encrypted_data, sensitive_fields)

        # Public data unchanged
        assert encrypted_data['public_disclosure'] == "Company acknowledges climate risks"

        # Confidential data encrypted then decrypted correctly
        assert decrypted_data['stakeholder_feedback'] == materiality_data['stakeholder_feedback']
        assert decrypted_data['confidential_impact_data'] == materiality_data['confidential_impact_data']


class TestEncryptionSecurity:
    """Security-focused tests."""

    def test_encrypted_data_is_not_plaintext(self):
        """Ensure encrypted data doesn't contain plaintext fragments."""
        key = Fernet.generate_key()
        em = EncryptionManager(key=key)

        sensitive_text = "CONFIDENTIAL_PASSWORD_123"
        encrypted = em.encrypt(sensitive_text)

        # Encrypted text should not contain any part of the original
        assert "CONFIDENTIAL" not in encrypted
        assert "PASSWORD" not in encrypted
        assert "123" not in encrypted

    def test_encryption_produces_different_output_each_time(self):
        """Test that encrypting the same data twice produces different ciphertext."""
        key = Fernet.generate_key()
        em = EncryptionManager(key=key)

        data = "same data"
        encrypted1 = em.encrypt(data)
        encrypted2 = em.encrypt(data)

        # Due to Fernet's use of timestamp and random IV, outputs should differ
        assert encrypted1 != encrypted2

        # But both should decrypt to the same value
        assert em.decrypt(encrypted1) == data
        assert em.decrypt(encrypted2) == data


# Run tests with: pytest tests/test_encryption.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
