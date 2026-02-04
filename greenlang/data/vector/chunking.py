"""
Document chunking service with multiple strategies.

Supports semantic, fixed-size, sliding window, and hierarchical
chunking strategies for embedding pipeline ingestion.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from greenlang.data.vector.models import ChunkResult, ChunkStrategy

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    strategy: ChunkStrategy = ChunkStrategy.SEMANTIC
    chunk_size: int = 512          # tokens
    chunk_overlap: int = 64        # tokens
    min_chunk_size: int = 50       # minimum tokens per chunk
    max_chunk_size: int = 1024     # maximum tokens per chunk
    separator: str = "\n\n"        # paragraph separator for semantic
    sentence_endings: str = r"[.!?]\s+"


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def _tokenize_approx(text: str, max_tokens: int) -> str:
    """Approximate tokenization by character count."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    # Try to cut at word boundary
    cut_point = text.rfind(" ", 0, max_chars)
    if cut_point == -1:
        cut_point = max_chars
    return text[:cut_point]


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks."""
        ...


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking: splits on sentence/paragraph boundaries.

    Preferred for regulations, legal docs, and structured text.
    Respects natural language boundaries for better embedding quality.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._sentence_re = re.compile(config.sentence_endings)

    def chunk(self, text: str) -> List[str]:
        # Split into paragraphs first
        paragraphs = text.split(self.config.separator)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _estimate_tokens(para)

            # If a single paragraph exceeds max, split by sentences
            if para_tokens > self.config.max_chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split paragraph into sentences
                sentences = self._sentence_re.split(para)
                sentences = [s.strip() for s in sentences if s.strip()]

                sent_chunk: List[str] = []
                sent_tokens = 0
                for sent in sentences:
                    st = _estimate_tokens(sent)
                    if sent_tokens + st > self.config.chunk_size and sent_chunk:
                        chunks.append(" ".join(sent_chunk))
                        # Overlap: keep last sentence
                        if self.config.chunk_overlap > 0 and sent_chunk:
                            overlap_text = sent_chunk[-1]
                            sent_chunk = [overlap_text]
                            sent_tokens = _estimate_tokens(overlap_text)
                        else:
                            sent_chunk = []
                            sent_tokens = 0
                    sent_chunk.append(sent)
                    sent_tokens += st

                if sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                continue

            # Check if adding paragraph exceeds chunk size
            if current_tokens + para_tokens > self.config.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Overlap: keep last paragraph
                if self.config.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text]
                    current_tokens = _estimate_tokens(overlap_text)
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return [c for c in chunks if _estimate_tokens(c) >= self.config.min_chunk_size]


class FixedChunker(ChunkingStrategy):
    """
    Fixed-size chunking: splits at exact token boundaries.

    Best for structured data and forms where semantic boundaries
    are less important than consistent chunk sizes.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(self, text: str) -> List[str]:
        chunks: List[str] = []
        char_size = self.config.chunk_size * 4
        char_overlap = self.config.chunk_overlap * 4
        step = max(1, char_size - char_overlap)

        pos = 0
        while pos < len(text):
            end = min(pos + char_size, len(text))
            chunk = text[pos:end].strip()
            if chunk and _estimate_tokens(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            pos += step

        return chunks


class SlidingWindowChunker(ChunkingStrategy):
    """
    Sliding window chunking: overlapping windows across text.

    Good for long narratives where context from adjacent
    windows helps maintain coherence.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(self, text: str) -> List[str]:
        # Split into words for more precise windowing
        words = text.split()
        window_words = self.config.chunk_size  # approximate words ~ tokens
        overlap_words = self.config.chunk_overlap

        if len(words) <= window_words:
            return [text.strip()] if text.strip() else []

        chunks: List[str] = []
        step = max(1, window_words - overlap_words)

        pos = 0
        while pos < len(words):
            end = min(pos + window_words, len(words))
            chunk = " ".join(words[pos:end]).strip()
            if chunk and _estimate_tokens(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            if end >= len(words):
                break
            pos += step

        return chunks


@dataclass
class HierarchicalChunk:
    text: str
    level: int  # 0 = document, 1 = section, 2 = paragraph
    parent_index: Optional[int] = None
    children: List[int] = field(default_factory=list)


class HierarchicalChunker(ChunkingStrategy):
    """
    Hierarchical chunking: creates parent-child chunk relationships.

    Best for complex documents with clear section structure
    (headings, subheadings). Enables multi-level retrieval.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def chunk(self, text: str) -> List[str]:
        """Returns flat list of chunks (use chunk_hierarchical for tree)."""
        tree = self.chunk_hierarchical(text)
        return [node.text for node in tree if _estimate_tokens(node.text) >= self.config.min_chunk_size]

    def chunk_hierarchical(self, text: str) -> List[HierarchicalChunk]:
        """Split into hierarchical chunks preserving parent-child relationships."""
        sections = self._split_by_headings(text)
        result: List[HierarchicalChunk] = []

        for section_text, level in sections:
            parent_idx = None
            # Find parent (nearest chunk with lower level)
            for i in range(len(result) - 1, -1, -1):
                if result[i].level < level:
                    parent_idx = i
                    break

            # If section is too large, sub-chunk it
            if _estimate_tokens(section_text) > self.config.max_chunk_size:
                sub_chunker = SemanticChunker(self.config)
                sub_chunks = sub_chunker.chunk(section_text)
                for sc in sub_chunks:
                    idx = len(result)
                    node = HierarchicalChunk(
                        text=sc, level=level, parent_index=parent_idx
                    )
                    result.append(node)
                    if parent_idx is not None:
                        result[parent_idx].children.append(idx)
            else:
                idx = len(result)
                node = HierarchicalChunk(
                    text=section_text, level=level, parent_index=parent_idx
                )
                result.append(node)
                if parent_idx is not None:
                    result[parent_idx].children.append(idx)

        return result

    def _split_by_headings(self, text: str) -> List[Tuple[str, int]]:
        """Split text by markdown headings, returning (text, level) pairs."""
        matches = list(self._heading_re.finditer(text))

        if not matches:
            return [(text, 0)]

        sections: List[Tuple[str, int]] = []

        # Content before first heading
        if matches[0].start() > 0:
            pre = text[: matches[0].start()].strip()
            if pre:
                sections.append((pre, 0))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((section_text, level))

        return sections


class ChunkingService:
    """
    Document chunking service with strategy selection.

    Provides a unified interface for all chunking strategies
    with automatic strategy selection based on document type.
    """

    STRATEGY_MAP = {
        ChunkStrategy.SEMANTIC: SemanticChunker,
        ChunkStrategy.FIXED: FixedChunker,
        ChunkStrategy.SLIDING_WINDOW: SlidingWindowChunker,
        ChunkStrategy.HIERARCHICAL: HierarchicalChunker,
    }

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk(
        self,
        text: str,
        strategy: Optional[ChunkStrategy] = None,
        config_override: Optional[ChunkingConfig] = None,
    ) -> ChunkResult:
        """
        Chunk text using the specified strategy.

        Args:
            text: The text to chunk
            strategy: Override the default strategy
            config_override: Override the default config
        """
        cfg = config_override or self.config
        strat = strategy or cfg.strategy

        chunker_cls = self.STRATEGY_MAP.get(strat)
        if not chunker_cls:
            raise ValueError(f"Unknown chunking strategy: {strat}")

        chunker = chunker_cls(cfg)
        chunks = chunker.chunk(text)

        return ChunkResult(
            chunks=chunks,
            chunk_count=len(chunks),
            strategy=strat.value,
            avg_chunk_size=sum(len(c) for c in chunks) // max(1, len(chunks)),
            metadata={
                "chunk_size_tokens": cfg.chunk_size,
                "chunk_overlap_tokens": cfg.chunk_overlap,
                "original_length": len(text),
                "original_tokens_est": _estimate_tokens(text),
            },
        )

    def chunk_document(
        self,
        text: str,
        source_type: str = "document",
    ) -> ChunkResult:
        """
        Auto-select chunking strategy based on source type.

        Strategy mapping:
        - regulation, policy -> semantic
        - document -> semantic
        - report -> sliding_window
        - emission_factor, benchmark -> fixed
        """
        strategy_map = {
            "regulation": ChunkStrategy.SEMANTIC,
            "policy": ChunkStrategy.SEMANTIC,
            "document": ChunkStrategy.SEMANTIC,
            "report": ChunkStrategy.SLIDING_WINDOW,
            "emission_factor": ChunkStrategy.FIXED,
            "benchmark": ChunkStrategy.FIXED,
        }
        strategy = strategy_map.get(source_type, ChunkStrategy.SEMANTIC)
        return self.chunk(text, strategy=strategy)

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n\n"
            doc.close()
            return text.strip()
        except ImportError:
            raise ImportError(
                "PyMuPDF required for PDF extraction. "
                "Install with: pip install PyMuPDF"
            )

    @staticmethod
    def extract_text_from_html(html: str) -> str:
        """Extract text from HTML content."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Fallback: strip HTML tags with regex
            clean = re.sub(r"<[^>]+>", "", html)
            return re.sub(r"\s+", " ", clean).strip()

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text for embedding."""
        # Normalize whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\t", " ", text)
        text = re.sub(r" +", " ", text)
        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
