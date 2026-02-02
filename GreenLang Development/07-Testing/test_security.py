# -*- coding: utf-8 -*-
"""
Security Tests
==============

Tests for GreenLang security features including:
- URL validation and HTTPS enforcement
- Path traversal protection
- Signature verification
- TLS configuration
"""

import os
import pytest
import tempfile
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Removed sys.path manipulation - using installed package
import sys

from greenlang.security import (
    validate_url,
    validate_git_url,
    validate_safe_path,
    safe_extract_tar,
    safe_extract_zip,
    PackVerifier,
    SignatureVerificationError,
    create_secure_session
)


class TestURLValidation:
    """Test URL validation and HTTPS enforcement"""

    def test_https_url_allowed(self):
        """Test that HTTPS URLs are allowed"""
        validate_url("https://example.com/file.tar.gz")
        validate_url("https://github.com/user/repo/archive.zip")

    def test_http_url_blocked(self):
        """Test that HTTP URLs are blocked by default"""
        with pytest.raises(ValueError, match="Insecure scheme 'http' not allowed"):
            validate_url("http://example.com/file.tar.gz")

    def test_http_url_allowed_in_dev_mode(self):
        """Test that HTTP URLs can be allowed in dev mode"""
        with patch.dict(os.environ, {"GL_ALLOW_INSECURE_FOR_DEV": "1"}):
            # Should not raise with dev flag
            validate_url("http://example.com/file.tar.gz")

    def test_invalid_scheme_blocked(self):
        """Test that invalid URL schemes are blocked"""
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("ftp://example.com/file.tar.gz")

        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("ssh://example.com/file.tar.gz")

    def test_git_url_validation(self):
        """Test Git URL validation"""
        # Valid HTTPS Git URL
        validate_git_url("https://github.com/user/repo.git")

        # Invalid HTTP Git URL
        with pytest.raises(ValueError, match="Only HTTPS Git repositories"):
            validate_git_url("http://github.com/user/repo.git")

        # Invalid SSH Git URL
        with pytest.raises(ValueError, match="Only HTTPS Git repositories"):
            validate_git_url("git@github.com:user/repo.git")

    def test_file_url_allowed_for_local(self):
        """Test that file:// URLs are allowed for local development"""
        validate_url("file:///path/to/local/file.tar.gz")


class TestPathTraversalProtection:
    """Test path traversal protection"""

    def test_safe_path_validation(self):
        """Test path validation against traversal"""
        base = Path("/safe/base")

        # Valid paths
        assert validate_safe_path(base, "subdir/file.txt")
        assert validate_safe_path(base, "./subdir/file.txt")

        # Invalid paths with traversal
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_safe_path(base, "../outside/file.txt")

        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_safe_path(base, "subdir/../../outside/file.txt")

    def test_tar_extraction_with_safe_paths(self):
        """Test safe tar extraction with normal paths"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a safe tar file
            tar_path = tmpdir / "safe.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                # Create a test file
                test_file = tmpdir / "test.txt"
                test_file.write_text("test content")
                tar.add(test_file, arcname="test.txt")

            # Extract safely
            extract_dir = tmpdir / "extract"
            safe_extract_tar(tar_path, extract_dir)

            # Verify file was extracted
            assert (extract_dir / "test.txt").exists()

    def test_tar_extraction_blocks_traversal(self):
        """Test that tar extraction blocks path traversal"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a malicious tar file with path traversal
            tar_path = tmpdir / "malicious.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                # Create a test file
                test_file = tmpdir / "evil.txt"
                test_file.write_text("evil content")

                # Add with traversal path
                info = tarfile.TarInfo(name="../../../etc/passwd")
                info.size = len(b"evil content")
                tar.addfile(info, test_file.open("rb"))

            # Attempt extraction should fail
            extract_dir = tmpdir / "extract"
            with pytest.raises(ValueError, match="Unsafe path in tar archive"):
                safe_extract_tar(tar_path, extract_dir)

    def test_tar_extraction_blocks_absolute_paths(self):
        """Test that tar extraction blocks absolute paths"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create tar with absolute path
            tar_path = tmpdir / "absolute.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                test_file = tmpdir / "test.txt"
                test_file.write_text("test")

                info = tarfile.TarInfo(name="/etc/passwd")
                info.size = 4
                tar.addfile(info, test_file.open("rb"))

            extract_dir = tmpdir / "extract"
            with pytest.raises(ValueError, match="Absolute path in archive not allowed"):
                safe_extract_tar(tar_path, extract_dir)

    def test_zip_extraction_with_safe_paths(self):
        """Test safe zip extraction with normal paths"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a safe zip file
            zip_path = tmpdir / "safe.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                test_file = tmpdir / "test.txt"
                test_file.write_text("test content")
                zf.write(test_file, arcname="test.txt")

            # Extract safely
            extract_dir = tmpdir / "extract"
            safe_extract_zip(zip_path, extract_dir)

            # Verify file was extracted
            assert (extract_dir / "test.txt").exists()

    def test_zip_extraction_blocks_traversal(self):
        """Test that zip extraction blocks path traversal"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create malicious zip with path traversal
            zip_path = tmpdir / "malicious.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                # Add file with traversal path
                zf.writestr("../../../etc/passwd", "evil content")

            extract_dir = tmpdir / "extract"
            with pytest.raises(ValueError, match="Unsafe path in zip archive"):
                safe_extract_zip(zip_path, extract_dir)


class TestSignatureVerification:
    """Test pack signature verification"""

    def test_verifier_initialization(self):
        """Test PackVerifier initialization"""
        verifier = PackVerifier()
        assert verifier is not None

        # Check if unsigned is blocked by default
        assert not verifier.allow_unsigned

    def test_verifier_with_dev_mode(self):
        """Test PackVerifier in dev mode"""
        with patch.dict(os.environ, {"GL_ALLOW_UNSIGNED_FOR_DEV": "1"}):
            verifier = PackVerifier()
            assert verifier.allow_unsigned

    def test_verify_unsigned_pack_blocked(self):
        """Test that unsigned packs are blocked by default"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create pack directory without signature
            pack_dir = tmpdir / "test-pack"
            pack_dir.mkdir()
            (pack_dir / "pack.yaml").write_text("name: test-pack\nversion: 1.0.0")

            verifier = PackVerifier()

            # Should raise error for unsigned pack
            with pytest.raises(SignatureVerificationError, match="No signature found"):
                verifier.verify_pack(pack_dir, require_signature=True)

    def test_verify_unsigned_pack_allowed_in_dev(self):
        """Test that unsigned packs can be allowed in dev mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create pack directory without signature
            pack_dir = tmpdir / "test-pack"
            pack_dir.mkdir()
            (pack_dir / "pack.yaml").write_text("name: test-pack\nversion: 1.0.0")

            with patch.dict(os.environ, {"GL_ALLOW_UNSIGNED_FOR_DEV": "1"}):
                verifier = PackVerifier()

                # Should not raise in dev mode
                verified, metadata = verifier.verify_pack(pack_dir, require_signature=True)
                assert not verified  # Not verified but allowed
                assert not metadata["signed"]

    def test_create_and_verify_stub_signature(self):
        """Test creating and verifying stub signatures"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create pack directory
            pack_dir = tmpdir / "test-pack"
            pack_dir.mkdir()
            (pack_dir / "pack.yaml").write_text("name: test-pack\nversion: 1.0.0")

            verifier = PackVerifier()

            # Create stub signature
            sig_path = verifier.create_signature_stub(pack_dir, publisher="test-publisher")
            assert sig_path.exists()

            # Verify with signature
            verified, metadata = verifier.verify_pack(pack_dir)
            assert metadata["signed"]
            assert metadata["publisher"] == "test-publisher"

    def test_checksum_verification(self):
        """Test checksum calculation and verification"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test file
            test_file = tmpdir / "test.txt"
            test_file.write_text("test content")

            verifier = PackVerifier()

            # Calculate checksum
            checksum1 = verifier._calculate_file_checksum(test_file)

            # Modify file
            test_file.write_text("modified content")
            checksum2 = verifier._calculate_file_checksum(test_file)

            # Checksums should differ
            assert checksum1 != checksum2


class TestTLSConfiguration:
    """Test TLS configuration and secure sessions"""

    def test_secure_session_creation(self):
        """Test that secure sessions are created properly"""
        session = create_secure_session()

        # Check that HTTPS adapter is configured
        assert "https://" in session.adapters

        # Check that HTTP is not configured by default
        if os.environ.get("GL_ALLOW_INSECURE_FOR_DEV") != "1":
            assert "http://" not in session.adapters

    def test_secure_session_with_ca_bundle(self):
        """Test secure session with custom CA bundle"""
        with tempfile.NamedTemporaryFile(suffix=".pem") as ca_file:
            ca_file.write(b"# Dummy CA cert")
            ca_file.flush()

            with patch.dict(os.environ, {"GL_CA_BUNDLE": ca_file.name}):
                # Should not raise
                session = create_secure_session()
                assert session is not None

    def test_secure_session_with_invalid_ca_bundle(self):
        """Test secure session with non-existent CA bundle"""
        with patch.dict(os.environ, {"GL_CA_BUNDLE": "/nonexistent/ca.pem"}):
            # Should warn but not fail
            session = create_secure_session()
            assert session is not None


class TestEndToEndSecurity:
    """End-to-end security tests"""

    @patch("greenlang.security.network.requests.Session.get")
    def test_install_blocks_http_url(self, mock_get):
        """Test that installer blocks HTTP URLs"""
        from greenlang.packs.installer import PackInstaller

        installer = PackInstaller()

        # Should reject HTTP URL
        with pytest.raises(ValueError, match="Insecure scheme 'http' not allowed"):
            installer.install("http://example.com/pack.tar.gz")

    @patch("greenlang.security.network.requests.Session.get")
    def test_install_allows_https_url(self, mock_get):
        """Test that installer allows HTTPS URLs"""
        from greenlang.packs.installer import PackInstaller

        # Mock response
        mock_response = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[b"test"])
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a mock pack archive
            pack_dir = tmpdir / "pack"
            pack_dir.mkdir()
            (pack_dir / "pack.yaml").write_text("name: test\nversion: 1.0.0")

            archive = tmpdir / "pack.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(pack_dir, arcname=".")

            with patch("greenlang.security.network.safe_download") as mock_download:
                with patch("greenlang.packs.installer.safe_extract_archive") as mock_extract:
                    with patch("greenlang.packs.installer.validate_pack_structure"):
                        with patch("greenlang.packs.installer.PackVerifier.verify_pack") as mock_verify:
                            mock_verify.return_value = (True, {"verified": True})

                            # Mock the download to use our test archive
                            def download_side_effect(url, dest, **kwargs):
                                import shutil
                                shutil.copy(archive, dest)

                            mock_download.side_effect = download_side_effect

                            # Mock extraction
                            def extract_side_effect(arch, dest):
                                dest.mkdir(exist_ok=True)
                                (dest / "pack.yaml").write_text("name: test\nversion: 1.0.0")

                            mock_extract.side_effect = extract_side_effect

                            installer = PackInstaller()

                            # Should work with HTTPS
                            # Note: This will still fail but for different reasons (pack structure)
                            # The important thing is it doesn't fail on URL validation
                            try:
                                installer.install("https://example.com/pack.tar.gz")
                            except Exception as e:
                                # Should not be URL validation error
                                assert "Insecure scheme" not in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])