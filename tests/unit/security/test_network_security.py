"""
Unit tests for network security (TLS/SSL enforcement)

SECURITY GATE: These tests verify that insecure network connections
are blocked by default and require explicit override.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from greenlang.registry.oci_client import OCIClient, OCIAuth, create_client


class TestNetworkSecurity:
    """Test suite for network security enforcement"""

    def test_http_url_install_fails(self):
        """Test I: HTTP URL install ⇒ fails by default"""
        # Try to create client with HTTP URL
        with pytest.raises(ValueError) as exc_info:
            client = OCIClient(registry="http://insecure.registry.com")

        assert "HTTP" in str(exc_info.value)
        assert "disabled by default" in str(exc_info.value)
        assert "GL_DEBUG_INSECURE" in str(exc_info.value)

    def test_https_with_bad_cert_fails(self):
        """Test J: HTTPS with bad cert ⇒ fails"""
        # Create client with HTTPS (should work)
        client = OCIClient(registry="https://secure.registry.com")

        # Mock SSL verification to fail
        import urllib.error
        import ssl

        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError(
                ssl.SSLError("certificate verify failed")
            )

            # Try to make a request
            with pytest.raises(urllib.error.URLError):
                client._make_request("GET", "https://secure.registry.com/v2/")

    def test_https_good_cert_succeeds(self):
        """Test K: HTTPS good cert ⇒ succeeds"""
        # Create client with HTTPS
        client = OCIClient(registry="https://ghcr.io")

        # Should not raise an exception
        assert client.registry == "https://ghcr.io"
        assert client.insecure == False

    @patch.dict(os.environ, {'GL_DEBUG_INSECURE': '1'})
    def test_http_with_insecure_flag_and_env(self):
        """Test L: HTTP with --insecure-transport and GL_DEBUG_INSECURE=1 ⇒ warn + allow"""
        import logging

        with patch.object(logging.getLogger('greenlang.registry.oci_client'), 'warning') as mock_warn:
            # Should work with both env var and flag
            client = OCIClient(
                registry="http://insecure.registry.com",
                insecure_transport=True
            )

            # Check that warning was logged
            assert mock_warn.called
            warning_msg = str(mock_warn.call_args[0][0])
            assert "SECURITY WARNING" in warning_msg
            assert "insecure" in warning_msg.lower()

    def test_insecure_tls_requires_env_var(self):
        """Test: Insecure TLS requires GL_DEBUG_INSECURE=1"""
        # Try to disable TLS verification without env var
        with pytest.raises(ValueError) as exc_info:
            client = OCIClient(registry="https://registry.com", insecure=True)

        assert "GL_DEBUG_INSECURE" in str(exc_info.value)
        assert "disabled by default" in str(exc_info.value)

    @patch.dict(os.environ, {'GL_DEBUG_INSECURE': '1'})
    def test_insecure_tls_with_env_var_warns(self):
        """Test: Insecure TLS with env var logs warning"""
        import logging

        with patch.object(logging.getLogger('greenlang.registry.oci_client'), 'warning') as mock_warn:
            client = OCIClient(registry="https://registry.com", insecure=True)

            # Check multiple warnings
            assert mock_warn.call_count >= 2
            warnings = [str(call[0][0]) for call in mock_warn.call_args_list]
            assert any("SECURITY WARNING" in w for w in warnings)
            assert any("SSL/TLS verification disabled" in w for w in warnings)
            assert any("MITM" in w for w in warnings)

    def test_https_prepended_when_no_protocol(self):
        """Test: HTTPS is prepended when no protocol specified"""
        client = OCIClient(registry="ghcr.io")
        assert client.registry == "https://ghcr.io"

        client2 = OCIClient(registry="docker.io")
        assert client2.registry == "https://docker.io"

    def test_create_client_enforces_https(self):
        """Test: create_client helper enforces HTTPS"""
        # Without protocol - should add https://
        client = create_client(registry="ghcr.io")
        assert client.registry == "https://ghcr.io"

        # With HTTP - should fail
        with pytest.raises(ValueError) as exc_info:
            client = create_client(registry="http://insecure.com")

        assert "HTTP" in str(exc_info.value)
        assert "disabled" in str(exc_info.value)

    @patch.dict(os.environ, {}, clear=True)
    def test_insecure_requires_both_env_and_flag(self):
        """Test: Insecure mode requires BOTH env var AND flag"""
        # Flag without env var - should fail
        with pytest.raises(ValueError):
            client = OCIClient(registry="https://registry.com", insecure=True)

        # Env var without flag - should not be insecure
        with patch.dict(os.environ, {'GL_DEBUG_INSECURE': '1'}):
            client = OCIClient(registry="https://registry.com", insecure=False)
            assert client.insecure == False

        # Both env var and flag - should work but warn
        with patch.dict(os.environ, {'GL_DEBUG_INSECURE': '1'}):
            import logging
            with patch.object(logging.getLogger('greenlang.registry.oci_client'), 'warning') as mock_warn:
                client = OCIClient(registry="https://registry.com", insecure=True)
                assert client.insecure == True
                assert mock_warn.called


class TestNetworkAllowlists:
    """Test network domain allowlisting"""

    def test_pack_with_wildcard_domain_rejected(self):
        """Test: Pack with wildcard domain '*' is rejected"""
        from greenlang.packs.installer import PackInstaller

        installer = PackInstaller()

        # Create capabilities with wildcard domain
        capabilities = {
            "net": {
                "allow": True,
                "outbound": {
                    "allowlist": ["*"]  # Too permissive
                }
            }
        }

        issues = installer._validate_capabilities(Mock(
            net=Mock(allow=True, outbound={"allowlist": ["*"]}),
            fs=None,
            clock=None,
            subprocess=None
        ))

        assert any("Wildcard '*' domain is too permissive" in issue for issue in issues)

    def test_pack_with_http_in_allowlist_rejected(self):
        """Test: Pack with HTTP URL in allowlist is rejected"""
        from greenlang.packs.installer import PackInstaller

        installer = PackInstaller()

        issues = installer._validate_capabilities(Mock(
            net=Mock(allow=True, outbound={"allowlist": ["http://insecure.com"]}),
            fs=None,
            clock=None,
            subprocess=None
        ))

        assert any("Insecure HTTP in allowlist" in issue for issue in issues)

    def test_pack_with_https_domains_accepted(self):
        """Test: Pack with HTTPS domains in allowlist is accepted"""
        from greenlang.packs.installer import PackInstaller

        installer = PackInstaller()

        issues = installer._validate_capabilities(Mock(
            net=Mock(allow=True, outbound={
                "allowlist": [
                    "api.openai.com",
                    "github.com",
                    "*.googleapis.com"
                ]
            }),
            fs=None,
            clock=None,
            subprocess=None
        ))

        # Should not have issues about these domains
        assert not any("api.openai.com" in issue for issue in issues)
        assert not any("github.com" in issue for issue in issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])