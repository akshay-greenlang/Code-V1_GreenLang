"""
Deterministic hashing for RAG components.

CRITICAL: This module provides canonical text normalization and stable hashing
for reproducible chunk IDs, document hashes, and section hashes.

Security: All hashes use SHA-256 for integrity verification.
Determinism: Unicode normalization (NFKC), line ending normalization, BOM removal.
"""

import hashlib
import unicodedata
import uuid
from typing import Dict


# Fixed namespace UUID for chunk ID generation (DNS namespace per RFC 4122)
CHUNK_NAMESPACE_UUID = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def sha256_str(s: str) -> str:
    """
    Compute SHA-256 hash of a string.

    Args:
        s: Input string

    Returns:
        Hexadecimal SHA-256 hash

    Example:
        >>> sha256_str("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    """
    Compute SHA-256 hash of bytes.

    Args:
        b: Input bytes

    Returns:
        Hexadecimal SHA-256 hash
    """
    return hashlib.sha256(b).hexdigest()


def canonicalize_text(s: str) -> str:
    """
    Canonicalize text for deterministic hashing.

    This function ensures that semantically equivalent text produces the same hash
    regardless of encoding differences, whitespace variations, or Unicode normalization.

    Operations:
    1. Unicode normalization (NFKC) - handles ligatures, compatibility characters
    2. Line ending normalization (CRLF → LF)
    3. BOM removal (UTF-8 byte order mark)
    4. Non-breaking space normalization (U+00A0 → U+0020)
    5. Em-space normalization (U+2003 → U+0020)
    6. Zero-width character removal (joiners, non-joiners, spaces)
    7. Lowercase conversion (case-insensitive comparison)
    8. Whitespace collapse (multiple spaces → single space)
    9. Strip leading/trailing whitespace

    Args:
        s: Input text

    Returns:
        Canonicalized text suitable for hashing

    Example:
        >>> canonicalize_text("Hello\r\n  World\u00a0!")  # CRLF, extra spaces, NBSP
        'hello world !'
        >>> canonicalize_text("fi")  # ligature fi (U+FB01)
        'fi'  # decomposed to f + i
    """
    # 1. Unicode normalization to NFKC (compatibility decomposition + canonical composition)
    # This handles ligatures (fi → f+i), superscripts, subscripts, etc.
    s = unicodedata.normalize("NFKC", s)

    # 2. Line ending normalization (Windows CRLF, Mac CR, Unix LF → LF)
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Remove BOM (byte order mark) U+FEFF
    s = s.lstrip("\ufeff")

    # 4. Normalize non-breaking spaces to regular spaces
    s = s.replace("\u00a0", " ")  # Non-breaking space (NBSP)
    s = s.replace("\u2003", " ")  # Em-space
    s = s.replace("\u2002", " ")  # En-space
    s = s.replace("\u2009", " ")  # Thin space

    # 5. Remove zero-width characters (used in some Unicode attacks)
    zero_width_chars = [
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u200e",  # Left-to-right mark
        "\u200f",  # Right-to-left mark
        "\u202a",  # Left-to-right embedding
        "\u202b",  # Right-to-left embedding
        "\u202c",  # Pop directional formatting
        "\u202d",  # Left-to-right override
        "\u202e",  # Right-to-left override
        "\ufeff",  # Zero-width non-breaking space (BOM)
    ]
    for char in zero_width_chars:
        s = s.replace(char, "")

    # 6. Lowercase for case-insensitive comparison
    s = s.lower()

    # 7. Collapse multiple whitespace characters to single space
    s = " ".join(s.split())

    # 8. Strip leading/trailing whitespace
    return s.strip()


def doc_hash(canonical_text: str, metadata: Dict[str, str]) -> str:
    """
    Compute deterministic hash of a document.

    Args:
        canonical_text: Canonicalized full document text
        metadata: Document metadata (title, version, publisher, etc.)

    Returns:
        SHA-256 hash of document (for version verification)

    Example:
        >>> text = canonicalize_text("Document content...")
        >>> meta = {"title": "GHG Protocol", "version": "1.05"}
        >>> doc_hash(text, meta)
        'a3f5b2c8d1e6f9a4b7c2d5e8f1a4b7c2...'
    """
    # Sort metadata keys for deterministic ordering
    sorted_meta = sorted(metadata.items())
    meta_str = "\n".join(f"{k}={v}" for k, v in sorted_meta)

    # Combine canonical text with sorted metadata
    combined = f"{canonical_text}\n<<METADATA>>\n{meta_str}"
    return sha256_str(combined)


def section_hash(section_text: str, section_path: str) -> str:
    """
    Compute deterministic hash of a document section.

    Args:
        section_text: Canonicalized section text
        section_path: Hierarchical section path (e.g., "Chapter 7 > 7.3.1")

    Returns:
        SHA-256 hash of section (for citation verification)

    Example:
        >>> text = canonicalize_text("Emission factors for stationary combustion...")
        >>> path = "Chapter 7 > Section 7.3 > 7.3.1"
        >>> section_hash(text, path)
        'b2c8d1e6...'
    """
    canonical = canonicalize_text(section_text)
    combined = f"{canonical}\n<<SECTION_PATH>>\n{section_path}"
    return sha256_str(combined)


def chunk_uuid5(doc_id: str, section_path: str, start_offset: int) -> str:
    """
    Generate deterministic UUID for a document chunk.

    Uses UUID v5 (name-based, SHA-1) with fixed namespace for reproducibility.

    Args:
        doc_id: Document identifier (UUID)
        section_path: Hierarchical section path
        start_offset: Character offset where chunk starts

    Returns:
        UUID v5 string (stable across runs)

    Example:
        >>> chunk_uuid5("doc123", "Ch7 > 7.3.1", 1024)
        'a3f5b2c8-d1e6-5f9a-8b7c-2d5e8f1a4b7c'

    Note:
        - Same inputs ALWAYS produce same UUID
        - Format: "{doc_id}|{section_path}|{start_offset}"
        - Encoded as UTF-8 before hashing
    """
    # Create deterministic name string
    name = f"{doc_id}|{section_path}|{start_offset}"

    # Generate UUID v5 using fixed namespace
    chunk_uuid = uuid.uuid5(CHUNK_NAMESPACE_UUID, name)

    return str(chunk_uuid)


def file_hash(file_path: str) -> str:
    """
    Compute SHA-256 hash of a file (for source verification).

    Args:
        file_path: Path to file

    Returns:
        SHA-256 hash of file contents

    Example:
        >>> file_hash("documents/ghg_protocol.pdf")
        'a3f5b2c8d1e6f9a4b7c2d5e8f1a4b7c2...'
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in 64KB chunks for memory efficiency
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def embedding_hash(embedding_vector: list) -> str:
    """
    Compute deterministic hash of an embedding vector.

    Used for verifying embedding consistency in replay mode.

    Args:
        embedding_vector: List of floats (embedding)

    Returns:
        SHA-256 hash of embedding

    Example:
        >>> vec = [0.123, 0.456, 0.789]
        >>> embedding_hash(vec)
        'b2c8d1e6...'

    Note:
        Floating-point precision differences may cause hash mismatches
        across different platforms (CPU vs GPU). Use with caution.
    """
    # Convert to bytes (assumes float64)
    import numpy as np
    vec_bytes = np.array(embedding_vector, dtype=np.float32).tobytes()
    return sha256_bytes(vec_bytes)


def query_hash(query: str, params: Dict) -> str:
    """
    Compute deterministic hash of a query + parameters (for caching).

    Args:
        query: Query string
        params: Query parameters (top_k, fetch_k, collections, etc.)

    Returns:
        SHA-256 hash of query + params

    Example:
        >>> query_hash("climate change", {"top_k": 5, "collections": ["ipcc"]})
        'c8d1e6f9...'
    """
    canonical_query = canonicalize_text(query)
    sorted_params = sorted(params.items())
    params_str = "\n".join(f"{k}={v}" for k, v in sorted_params)
    combined = f"{canonical_query}\n<<PARAMS>>\n{params_str}"
    return sha256_str(combined)
