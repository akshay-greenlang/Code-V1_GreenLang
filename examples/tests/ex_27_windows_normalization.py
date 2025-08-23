"""Example 27: Windows line-ending and path normalization."""

import pytest
import tempfile
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from examples.utils.normalizers import normalize_text

@pytest.mark.example
def test_windows_line_ending_normalization():
    """Normalize Windows (CRLF) to Unix (LF) line endings."""
    # Text with different line endings
    windows_text = "Line 1\r\nLine 2\r\nLine 3"
    unix_text = "Line 1\nLine 2\nLine 3"
    mixed_text = "Line 1\r\nLine 2\nLine 3"
    
    # All should normalize to same result
    assert normalize_text(windows_text) == normalize_text(unix_text)
    assert normalize_text(mixed_text) == normalize_text(unix_text)

@pytest.mark.example
def test_windows_path_normalization():
    """Normalize Windows paths in output."""
    # Windows paths with backslashes
    text_with_paths = """
    Loading file from C:\\Users\\test\\Documents\\data.json
    Saving to D:\\Projects\\GreenLang\\output.csv
    Temp file at C:\\Temp\\file123.tmp
    """
    
    normalized = normalize_text(text_with_paths)
    
    # Windows paths should be replaced with placeholder
    assert "C:\\Users" not in normalized
    assert "D:\\Projects" not in normalized
    assert "<PATH>" in normalized

@pytest.mark.example
def test_cross_platform_file_operations():
    """File operations work across platforms."""
    # Create temp file with platform-specific path
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name
        # Write with Windows line endings
        f.write("Line 1\r\n")
        f.write("Line 2\r\n")
    
    try:
        # Read file
        with open(temp_path, 'r') as f:
            content = f.read()
        
        # Normalize for comparison
        normalized = content.replace('\r\n', '\n').replace('\r', '\n')
        assert normalized == "Line 1\nLine 2\n"
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@pytest.mark.example
def test_path_separator_handling():
    """Handle both forward and backward slashes."""
    # Test path joining
    parts = ["examples", "fixtures", "data.json"]
    
    # Use os.path.join for platform-specific separator
    path = os.path.join(*parts)
    
    # Should work on any platform
    assert "examples" in path
    assert "fixtures" in path
    assert "data.json" in path
    
    # Normalize paths for comparison
    unix_style = "/".join(parts)
    windows_style = "\\".join(parts)
    
    # Both should refer to same file
    assert os.path.normpath(unix_style) == os.path.normpath(windows_style)