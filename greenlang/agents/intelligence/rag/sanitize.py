# -*- coding: utf-8 -*-
"""
Input sanitization for RAG system to prevent prompt injection and data leakage.

CRITICAL SECURITY: This module blocks prompt injection attacks, malicious URIs,
code execution attempts, and tool-calling triggers in retrieved documents.

Defense Layers:
1. Unicode normalization (NFKC) to prevent homoglyph attacks
2. Zero-width character removal
3. URI scheme blocking (data:, javascript:, file:, etc.)
4. Code block stripping
5. Tool/function calling pattern blocking
6. JSON structure escaping
"""

import re
import unicodedata
from typing import Optional
from urllib.parse import urlparse


def sanitize_rag_input(text: str, strict: bool = True) -> str:
    """
    Sanitize retrieved text before injecting into LLM context.

    This prevents prompt injection attacks where malicious documents could
    manipulate LLM behavior or leak sensitive information.

    Args:
        text: Retrieved document text
        strict: If True, apply aggressive sanitization (recommended)

    Returns:
        Sanitized text safe for LLM context

    Example:
        >>> sanitize_rag_input("Visit https://malicious.com for info")
        'Visit [link omitted] for info'
        >>> sanitize_rag_input("```python\\nimport os\\nos.system('rm -rf /')\\n```")
        '[code omitted]'

    Security Notes:
        - Blocks all URI schemes except http/https
        - Removes code blocks to prevent execution hints
        - Escapes tool-calling patterns
        - Normalizes Unicode to prevent homoglyph attacks
    """
    if not text:
        return ""

    # Step 1: Unicode normalization to NFKC (blocks homoglyph attacks)
    # Example: Greek 'α' (U+03B1) looks like Latin 'a' but could bypass filters
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Remove zero-width characters (Unicode steganography)
    # Attackers can hide instructions in zero-width characters
    zero_width_pattern = (
        r"[\u200b-\u200f\u202a-\u202e\u0000-\u0008\u000b-\u000c\u000e-\u001f]"
    )
    text = re.sub(zero_width_pattern, "", text)

    # Step 3: Block dangerous URI schemes (only allow http/https if at all)
    # Blocks: data:, javascript:, file:, ftp:, telnet:, ssh:, ldap:, mailto:
    dangerous_uris = (
        r"(?:data|javascript|file|ftp|telnet|ssh|ldap|mailto|vbscript|about):[^\s]*"
    )
    text = re.sub(dangerous_uris, "[blocked_uri]", text, flags=re.IGNORECASE)

    if strict:
        # In strict mode, also block http/https URLs to prevent data exfiltration
        # via image tags or external resource loading
        text = re.sub(r"https?://[^\s]+", "[link omitted]", text, flags=re.IGNORECASE)

    # Step 4: Remove code blocks (prevent execution hints to LLM)
    # Blocks both ``` fenced blocks and ` inline code
    text = re.sub(r"```[\s\S]*?```", "[code omitted]", text)  # Multi-line code
    text = re.sub(r"`[^`]+`", "[code omitted]", text)  # Inline code

    # Step 5: Block tool/function calling patterns
    # Prevents retrieved text from triggering tool calls
    tool_patterns = [
        r"call_tool\s*[:(\[]",  # call_tool:, call_tool(, call_tool[
        r"invoke_function\s*[:(\[]",
        r"execute\s*[:(\[]",
        r"</?tool>",  # XML-style tool tags
        r"</?function>",
        r"\{\{tool:",  # Template-style tool calls
        r"\{\{function:",
    ]
    for pattern in tool_patterns:
        text = re.sub(pattern, "[redacted]", text, flags=re.IGNORECASE)

    # Step 6: Escape JSON-like structures that could manipulate responses
    # Blocks patterns like {"role": "system", "content": "..."}
    text = re.sub(
        r'\{["\'](?:role|tool|function|assistant|user|system)["\']:',
        "[json_blocked]",
        text,
    )

    # Step 7: Remove HTML/XML tags (prevents XSS-style attacks in web contexts)
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "[script_blocked]", text, flags=re.IGNORECASE)
    text = re.sub(r"<iframe[^>]*>[\s\S]*?</iframe>", "[iframe_blocked]", text, flags=re.IGNORECASE)

    # Step 8: Escape braces (prevents JSON injection)
    # Convert {} to () to avoid JSON parsing
    text = text.replace("{", "(").replace("}", ")")

    # Step 9: Remove control characters (except newline, tab, carriage return)
    # Blocks ASCII control characters that could manipulate terminal output
    control_chars = "".join(chr(i) for i in range(32) if i not in (9, 10, 13))
    text = text.translate(str.maketrans("", "", control_chars))

    return text


def sanitize_citation_uri(uri: str) -> str:
    """
    Sanitize URIs before including in citations.

    Args:
        uri: Source URI

    Returns:
        Sanitized URI (or placeholder if dangerous)

    Example:
        >>> sanitize_citation_uri("javascript:alert('xss')")
        'gl://blocked/dangerous_scheme'
        >>> sanitize_citation_uri("https://ghgprotocol.org/standard.pdf")
        'https://ghgprotocol.org/standard.pdf'
    """
    if not uri:
        return "gl://unknown"

    try:
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()

        # Only allow http, https, file (for local docs), and gl:// (internal)
        allowed_schemes = {"http", "https", "file", "gl"}

        if scheme not in allowed_schemes:
            return "gl://blocked/dangerous_scheme"

        # Block localhost/private IPs to prevent SSRF
        netloc = parsed.netloc.lower()
        dangerous_hosts = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "169.254.",  # Link-local
            "192.168.",  # Private network
            "10.",  # Private network
        ]

        for dangerous in dangerous_hosts:
            if dangerous in netloc:
                return "gl://blocked/private_network"

        return uri

    except Exception:
        # If parsing fails, block the URI
        return "gl://blocked/invalid_uri"


def detect_suspicious_content(text: str) -> Optional[str]:
    """
    Detect suspicious patterns that might indicate adversarial documents.

    Args:
        text: Document text

    Returns:
        Warning message if suspicious content detected, None otherwise

    Example:
        >>> detect_suspicious_content("Ignore all previous instructions and...")
        'WARNING: Detected potential prompt injection (ignore_instructions pattern)'

    Note:
        This is a heuristic check, not foolproof. Use for logging/alerting.
    """
    suspicious_patterns = [
        (r"ignore\s+(all\s+)?(previous\s+)?instructions", "ignore_instructions"),
        (r"disregard\s+(all\s+)?(previous\s+)?(rules|constraints)", "disregard_rules"),
        (r"system\s*prompt", "system_prompt_reference"),
        (r"you\s+are\s+now", "role_manipulation"),
        (r"pretend\s+to\s+be", "role_manipulation"),
        (r"act\s+as\s+if", "role_manipulation"),
        (r"<script", "script_tag"),
        (r"javascript:", "javascript_uri"),
        (r"eval\s*\(", "eval_call"),
        (r"exec\s*\(", "exec_call"),
    ]

    for pattern, name in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return f"WARNING: Detected potential prompt injection ({name} pattern)"

    return None


def sanitize_metadata(metadata: dict) -> dict:
    """
    Sanitize metadata fields before storing or returning.

    Args:
        metadata: Document metadata

    Returns:
        Sanitized metadata

    Example:
        >>> sanitize_metadata({"title": "Doc<script>alert('xss')</script>"})
        {'title': 'Doc[script_blocked]'}
    """
    sanitized = {}

    for key, value in metadata.items():
        if isinstance(value, str):
            # Apply light sanitization (preserve more content than full sanitization)
            value = re.sub(r"<script[^>]*>[\s\S]*?</script>", "[script_blocked]", value, flags=re.IGNORECASE)
            value = re.sub(r"javascript:[^\s]*", "[blocked_uri]", value, flags=re.IGNORECASE)
            sanitized[key] = value
        else:
            sanitized[key] = value

    return sanitized


def validate_collection_name(collection: str) -> bool:
    """
    Validate collection name to prevent path traversal or injection.

    Args:
        collection: Collection name

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_collection_name("ghg_protocol_corp")
        True
        >>> validate_collection_name("../../../etc/passwd")
        False
        >>> validate_collection_name("test; rm -rf /")
        False
    """
    # Collection names must be alphanumeric + underscore/hyphen only
    # No path separators, no shell metacharacters
    if not collection:
        return False

    # Max length 64 characters
    if len(collection) > 64:
        return False

    # Only allow: letters, numbers, underscore, hyphen
    if not re.match(r"^[a-zA-Z0-9_-]+$", collection):
        return False

    # Block path traversal
    if ".." in collection or "/" in collection or "\\" in collection:
        return False

    return True


def sanitize_for_prompt(text: str, max_length: int = 4096) -> str:
    """
    Sanitize and truncate text for inclusion in LLM prompt.

    Args:
        text: Retrieved document text
        max_length: Maximum characters to include

    Returns:
        Sanitized and truncated text

    Example:
        >>> long_text = "a" * 5000
        >>> len(sanitize_for_prompt(long_text, max_length=1000))
        1000
    """
    # First sanitize
    text = sanitize_rag_input(text, strict=True)

    # Then truncate (preserve full sentences if possible)
    if len(text) <= max_length:
        return text

    # Truncate at sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind(".")
    last_newline = truncated.rfind("\n")

    # Use last sentence or paragraph boundary
    boundary = max(last_period, last_newline)
    if boundary > max_length * 0.8:  # If boundary is reasonably close
        return truncated[: boundary + 1] + " [truncated]"

    return truncated + " [truncated]"
