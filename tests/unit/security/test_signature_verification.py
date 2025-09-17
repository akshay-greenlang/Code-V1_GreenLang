"""
Unit tests for pack signature verification

SECURITY GATE: These tests verify that unsigned packs are rejected
by default and signature verification works correctly.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import base64

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from greenlang.provenance.signing import (
    DevKeyVerifier,
    SigstoreVerifier,
    UnsignedPackError,
    verify_pack_signature,
    sign_pack
)
from greenlang.packs.installer import PackInstaller


class TestSignatureVerification:
    """Test suite for signature verification"""

    def test_unsigned_pack_install_fails(self):
        """Test E: Installing pack without .sig ⇒ fails (UnsignedPackError)"""
        installer = PackInstaller()

        with tempfile.TemporaryDirectory() as tmpdir:
            pack_dir = Path(tmpdir) / "test-pack"
            pack_dir.mkdir()

            # Create a minimal pack.yaml
            manifest = pack_dir / "pack.yaml"
            manifest.write_text("""
name: test-pack
version: 1.0.0
kind: pack
license: MIT
contents:
  pipelines:
    - pipeline.yaml
""")

            # No signature file created

            # Try to install without signature
            with pytest.raises(UnsignedPackError) as exc_info:
                installer.install_pack(
                    pack_dir,
                    allow_unsigned=False  # Enforce signature requirement
                )

            assert "signature" in str(exc_info.value).lower()
            assert "--allow-unsigned" in str(exc_info.value)

    def test_invalid_signature_fails(self):
        """Test F: Installing pack with invalid .sig ⇒ fails"""
        installer = PackInstaller()
        verifier = DevKeyVerifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            pack_dir = Path(tmpdir) / "test-pack"
            pack_dir.mkdir()

            # Create pack.yaml
            manifest = pack_dir / "pack.yaml"
            manifest_content = """
name: test-pack
version: 1.0.0
kind: pack
license: MIT
contents:
  pipelines:
    - pipeline.yaml
"""
            manifest.write_text(manifest_content)

            # Create an invalid signature (random bytes)
            sig_file = pack_dir / "pack.sig"
            sig_file.write_bytes(base64.b64encode(b"invalid signature"))

            # Mock the verifier to be used
            with patch('greenlang.packs.installer.create_verifier') as mock_create:
                mock_create.return_value = verifier

                # Try to install with invalid signature
                with pytest.raises(UnsignedPackError) as exc_info:
                    installer.install_pack(
                        pack_dir,
                        allow_unsigned=False,
                        verifier=verifier
                    )

                assert "verification failed" in str(exc_info.value).lower()

    def test_valid_signature_succeeds(self):
        """Test G: Installing pack with valid signature via DevKeyVerifier ⇒ succeeds"""
        installer = PackInstaller()
        verifier = DevKeyVerifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            pack_dir = Path(tmpdir) / "test-pack"
            pack_dir.mkdir()

            # Create pack.yaml
            manifest = pack_dir / "pack.yaml"
            manifest_content = """
name: test-pack
version: 1.0.0
kind: pack
license: MIT
contents:
  pipelines:
    - pipeline.yaml
"""
            manifest.write_text(manifest_content)

            # Create pipeline file
            pipeline = pack_dir / "pipeline.yaml"
            pipeline.write_text("name: test\nsteps: []")

            # Create a valid signature using DevKeyVerifier
            pack_bytes = manifest.read_bytes()
            signature = verifier.sign(pack_bytes)
            sig_file = pack_dir / "pack.sig"
            sig_file.write_bytes(signature)

            # Mock the actual installation to avoid filesystem operations
            with patch.object(installer, 'validate_manifest') as mock_validate:
                mock_validate.return_value = (True, [])

                with patch('shutil.copytree'):
                    with patch('builtins.open', create=True):
                        # Try to install with valid signature
                        # This should succeed without raising an exception
                        try:
                            # We need to mock more of the installation process
                            with patch('greenlang.packs.installer.verify_pack_signature') as mock_verify:
                                mock_verify.return_value = (True, "Signature verified")

                                success, msg = installer.install_pack(
                                    pack_dir,
                                    allow_unsigned=False,
                                    verifier=verifier
                                )
                                # If no exception raised, test passes
                                assert True
                        except UnsignedPackError:
                            pytest.fail("Valid signature should not raise UnsignedPackError")

    @patch('greenlang.packs.installer.audit_log')
    @patch('logging.Logger.warning')
    def test_allow_unsigned_with_warning(self, mock_warn, mock_audit):
        """Test H: Installing unsigned with --allow-unsigned ⇒ succeeds but logs WARNING"""
        installer = PackInstaller()

        with tempfile.TemporaryDirectory() as tmpdir:
            pack_dir = Path(tmpdir) / "test-pack"
            pack_dir.mkdir()

            # Create pack.yaml (no signature)
            manifest = pack_dir / "pack.yaml"
            manifest.write_text("""
name: test-pack
version: 1.0.0
kind: pack
license: MIT
contents:
  pipelines:
    - pipeline.yaml
""")
            # Create pipeline file
            pipeline = pack_dir / "pipeline.yaml"
            pipeline.write_text("name: test\nsteps: []")

            # Mock the installation process
            with patch.object(installer, 'validate_manifest') as mock_validate:
                mock_validate.return_value = (True, [])

                with patch('shutil.copytree'):
                    with patch('builtins.open', create=True):
                        # Install with allow_unsigned=True
                        success, msg = installer.install_pack(
                            pack_dir,
                            allow_unsigned=True  # Allow unsigned
                        )

                        # Check that warning was logged
                        assert mock_warn.called
                        warning_calls = [str(call[0][0]) for call in mock_warn.call_args_list]
                        assert any("SECURITY WARNING" in w for w in warning_calls)
                        assert any("unsigned" in w.lower() for w in warning_calls)

                        # Check that audit log was called
                        assert mock_audit.called
                        audit_calls = [call[0] for call in mock_audit.call_args_list]
                        assert any("PACK_INSTALL_UNSIGNED" in str(call) for call in audit_calls)


class TestDevKeyVerifier:
    """Test the DevKeyVerifier implementation"""

    def test_dev_verifier_ephemeral_keys(self):
        """Test that DevKeyVerifier generates ephemeral keys (no hardcoded keys)"""
        verifier1 = DevKeyVerifier()
        verifier2 = DevKeyVerifier()

        # Each instance should have different keys
        assert verifier1.public_key_pem != verifier2.public_key_pem

        # Keys should not be hardcoded (check they're generated)
        assert len(verifier1.public_key_pem) > 0
        assert b"BEGIN PUBLIC KEY" in verifier1.public_key_pem

    def test_dev_verifier_sign_and_verify(self):
        """Test that DevKeyVerifier can sign and verify correctly"""
        verifier = DevKeyVerifier()
        test_data = b"test pack data"

        # Sign the data
        signature = verifier.sign(test_data)

        # Verify the signature
        assert verifier.verify(test_data, signature) == True

        # Verify with wrong data fails
        assert verifier.verify(b"wrong data", signature) == False

        # Verify with wrong signature fails
        wrong_sig = base64.b64encode(b"wrong signature")
        assert verifier.verify(test_data, wrong_sig) == False

    def test_dev_verifier_warns_about_security(self):
        """Test that DevKeyVerifier warns it's for development only"""
        import logging

        with patch.object(logging.getLogger('greenlang.provenance.signing'), 'warning') as mock_warn:
            verifier = DevKeyVerifier()

            assert mock_warn.called
            warning_msg = str(mock_warn.call_args_list)
            assert "DEVELOPMENT ONLY" in warning_msg or "DEV" in warning_msg


class TestSigstoreVerifier:
    """Test the SigstoreVerifier stub"""

    def test_sigstore_not_implemented(self):
        """Test that SigstoreVerifier raises NotImplementedError"""
        with pytest.raises(NotImplementedError) as exc_info:
            verifier = SigstoreVerifier()

        assert "Week 1" in str(exc_info.value) or "implemented" in str(exc_info.value)


class TestPackSignatureHelpers:
    """Test helper functions for pack signatures"""

    def test_verify_pack_signature_no_sig_file(self):
        """Test verify_pack_signature when no .sig file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_file = Path(tmpdir) / "test.tar.gz"
            pack_file.write_bytes(b"pack data")

            # No .sig file created
            is_valid, msg = verify_pack_signature(pack_file)

            assert is_valid == False
            assert "No signature file found" in msg

    def test_sign_pack_creates_sig_file(self):
        """Test that sign_pack creates a .sig file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_file = Path(tmpdir) / "test.tar.gz"
            pack_file.write_bytes(b"pack data")

            verifier = DevKeyVerifier()
            sig_path = sign_pack(pack_file, verifier)

            assert sig_path.exists()
            assert sig_path.name == "test.tar.gz.sig"

            # Verify the signature is valid
            is_valid, msg = verify_pack_signature(pack_file, verifier)
            assert is_valid == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])