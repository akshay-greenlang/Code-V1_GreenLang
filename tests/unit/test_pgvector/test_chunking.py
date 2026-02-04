"""Tests for pgvector chunking service."""

import pytest

from greenlang.data.vector.chunking import (
    ChunkingConfig,
    ChunkingService,
    FixedChunker,
    HierarchicalChunker,
    SemanticChunker,
    SlidingWindowChunker,
    _estimate_tokens,
)
from greenlang.data.vector.models import ChunkStrategy


class TestEstimateTokens:
    def test_basic_estimation(self):
        assert _estimate_tokens("") == 1  # minimum 1
        assert _estimate_tokens("word") == 1
        assert _estimate_tokens("a" * 100) == 25  # ~4 chars/token

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = _estimate_tokens(text)
        assert tokens > 50


class TestSemanticChunker:
    def setup_method(self):
        self.config = ChunkingConfig(
            strategy=ChunkStrategy.SEMANTIC,
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=10,
        )
        self.chunker = SemanticChunker(self.config)

    def test_simple_text(self):
        text = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(len(c) > 0 for c in chunks)

    def test_single_paragraph(self):
        text = "Short text."
        chunks = self.chunker.chunk(text)
        # May be empty if below min_chunk_size
        assert isinstance(chunks, list)

    def test_long_paragraph_splits_by_sentence(self):
        text = ("This is a long sentence. " * 100)
        chunks = self.chunker.chunk(text)
        assert len(chunks) > 1

    def test_preserves_content(self):
        text = "First paragraph about climate.\n\nSecond paragraph about emissions."
        chunks = self.chunker.chunk(text)
        combined = " ".join(chunks)
        assert "climate" in combined
        assert "emissions" in combined


class TestFixedChunker:
    def setup_method(self):
        self.config = ChunkingConfig(
            strategy=ChunkStrategy.FIXED,
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_size=5,
        )
        self.chunker = FixedChunker(self.config)

    def test_fixed_chunking(self):
        text = "word " * 200
        chunks = self.chunker.chunk(text)
        assert len(chunks) > 1

    def test_short_text(self):
        text = "Short."
        chunks = self.chunker.chunk(text)
        assert isinstance(chunks, list)


class TestSlidingWindowChunker:
    def setup_method(self):
        self.config = ChunkingConfig(
            strategy=ChunkStrategy.SLIDING_WINDOW,
            chunk_size=50,
            chunk_overlap=25,
            min_chunk_size=5,
        )
        self.chunker = SlidingWindowChunker(self.config)

    def test_sliding_window(self):
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = self.chunker.chunk(text)
        assert len(chunks) > 1

    def test_overlap_content(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = self.chunker.chunk(text)
        if len(chunks) >= 2:
            # Adjacent chunks should share some words
            words1 = set(chunks[0].split())
            words2 = set(chunks[1].split())
            # With 50% overlap, there should be significant intersection
            assert len(words1 & words2) > 0


class TestHierarchicalChunker:
    def setup_method(self):
        self.config = ChunkingConfig(
            strategy=ChunkStrategy.HIERARCHICAL,
            chunk_size=100,
            min_chunk_size=5,
        )
        self.chunker = HierarchicalChunker(self.config)

    def test_markdown_headings(self):
        text = """# Section 1
Content for section one.

## Subsection 1.1
Details for subsection.

# Section 2
Content for section two."""
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 2

    def test_hierarchical_tree(self):
        text = """# Title

Introduction text.

## Chapter 1

Chapter one content.

### Section 1.1

Section details.

## Chapter 2

Chapter two content."""
        tree = self.chunker.chunk_hierarchical(text)
        assert len(tree) >= 3
        # Check parent-child relationships
        has_parent = any(n.parent_index is not None for n in tree)
        assert has_parent

    def test_no_headings(self):
        text = "Plain text without any headings. Just paragraphs."
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 1


class TestChunkingService:
    def setup_method(self):
        self.service = ChunkingService()

    def test_default_strategy(self):
        result = self.service.chunk("Sample text for chunking. " * 50)
        assert result.strategy == "semantic"
        assert result.chunk_count > 0

    def test_fixed_strategy(self):
        result = self.service.chunk(
            "Sample text. " * 100,
            strategy=ChunkStrategy.FIXED,
        )
        assert result.strategy == "fixed"

    def test_chunk_document_auto_select(self):
        text = "Regulation content. " * 50
        result = self.service.chunk_document(text, source_type="regulation")
        assert result.strategy == "semantic"

        result = self.service.chunk_document(text, source_type="report")
        assert result.strategy == "sliding_window"

        result = self.service.chunk_document(text, source_type="emission_factor")
        assert result.strategy == "fixed"

    def test_chunk_metadata(self):
        result = self.service.chunk("Sample text for testing chunks. " * 20)
        assert "chunk_size_tokens" in result.metadata
        assert "original_length" in result.metadata

    def test_clean_text(self):
        text = "Line 1\r\nLine 2\t\ttabbed\n\n\n\nextra newlines"
        cleaned = ChunkingService.clean_text(text)
        assert "\r" not in cleaned
        assert "\t" not in cleaned
        assert "\n\n\n" not in cleaned

    def test_extract_text_from_html(self):
        html = "<html><body><h1>Title</h1><p>Content</p><script>alert('x')</script></body></html>"
        text = ChunkingService.extract_text_from_html(html)
        assert "Title" in text
        assert "Content" in text
        assert "alert" not in text
