# -*- coding: utf-8 -*-
"""
Token-aware chunking for RAG system.

Implements intelligent document chunking with:
- Token-based chunking (not character-based)
- Sentence boundary preservation
- Configurable chunk size and overlap
- Stable chunk IDs (UUID v5)
"""

import re
from typing import List, Optional
import logging

from greenlang.agents.intelligence.rag.models import Chunk
from greenlang.agents.intelligence.rag.hashing import chunk_uuid5, section_hash
from greenlang.agents.intelligence.rag.config import RAGConfig, get_config

logger = logging.getLogger(__name__)


class TokenAwareChunker:
    """
    Token-aware chunker using tiktoken or HuggingFace tokenizer.

    Features:
    - Token-based chunking (512 tokens default)
    - Sentence boundary awareness
    - Configurable overlap
    - Stable chunk IDs using UUID v5
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        tokenizer: str = "tiktoken",
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize token-aware chunker.

        Args:
            chunk_size: Chunk size in tokens (default: 512)
            overlap: Overlap in tokens (default: 64)
            tokenizer: Tokenizer to use ("tiktoken" or "huggingface")
            config: RAG configuration
        """
        self.config = config or get_config()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer_name = tokenizer

        # Initialize tokenizer
        self._tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        """Initialize the tokenizer."""
        if self.tokenizer_name == "tiktoken":
            try:
                import tiktoken

                # Use cl100k_base (GPT-4) tokenizer
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info("Initialized tiktoken tokenizer (cl100k_base)")

            except ImportError:
                logger.warning(
                    "tiktoken not installed, falling back to simple tokenizer. "
                    "Install with: pip install tiktoken"
                )
                self._tokenizer = None

        elif self.tokenizer_name == "huggingface":
            try:
                from transformers import AutoTokenizer

                # Use all-MiniLM-L6-v2 tokenizer (matches embedding model)
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Initialized HuggingFace tokenizer (all-MiniLM-L6-v2)")

            except ImportError:
                logger.warning(
                    "transformers not installed, falling back to simple tokenizer. "
                    "Install with: pip install transformers"
                )
                self._tokenizer = None

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if self._tokenizer is None:
            # Fallback: simple whitespace tokenization
            return text.split()

        if self.tokenizer_name == "tiktoken":
            return self._tokenizer.encode(text)
        elif self.tokenizer_name == "huggingface":
            return self._tokenizer.encode(text, add_special_tokens=False)
        else:
            return text.split()

    def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenize tokens back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Detokenized text
        """
        if self._tokenizer is None:
            # Fallback: join with spaces
            return " ".join(str(t) for t in tokens)

        if self.tokenizer_name == "tiktoken":
            return self._tokenizer.decode(tokens)
        elif self.tokenizer_name == "huggingface":
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            return " ".join(str(t) for t in tokens)

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Uses regex-based sentence splitting with support for:
        - Periods, exclamation marks, question marks
        - Abbreviations (Dr., Mr., etc.)
        - Decimals (1.5, 3.14)

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Regex pattern for sentence splitting
        # Matches: . ! ? followed by whitespace and capital letter
        # Excludes: abbreviations, decimals
        sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])"

        sentences = re.split(sentence_pattern, text)

        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        section_path: str = "Document",
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        Chunk a document into token-aware chunks.

        Algorithm:
        1. Split document into sentences
        2. Group sentences into chunks of ~chunk_size tokens
        3. Add overlap between chunks
        4. Generate stable chunk IDs using UUID v5

        Args:
            text: Document text
            doc_id: Document ID
            section_path: Section path (e.g., "Chapter 7 > 7.3.1")
            page_start: Starting page number
            page_end: Ending page number
            extra: Extra metadata

        Returns:
            List of Chunk objects

        Example:
            >>> chunker = TokenAwareChunker(chunk_size=512, overlap=64)
            >>> chunks = chunker.chunk_document(
            ...     text="Long document text...",
            ...     doc_id="doc123",
            ...     section_path="Chapter 1 > Introduction"
            ... )
            >>> len(chunks)
            3
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunker")
            return []

        # Split into sentences
        sentences = self.split_sentences(text)

        if not sentences:
            # Fallback: treat entire text as one sentence
            sentences = [text]

        logger.debug(f"Split text into {len(sentences)} sentences")

        # Build chunks by grouping sentences
        chunks = []
        current_chunk_tokens = []
        current_chunk_sentences = []
        current_start_char = 0
        char_offset = 0

        for sentence in sentences:
            # Tokenize sentence
            sentence_tokens = self.tokenize(sentence)

            # Check if adding this sentence exceeds chunk size
            if current_chunk_tokens and len(current_chunk_tokens) + len(sentence_tokens) > self.chunk_size:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_chunk_sentences)
                chunk_end_char = char_offset

                chunk = self._create_chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    section_path=section_path,
                    start_char=current_start_char,
                    end_char=chunk_end_char,
                    token_count=len(current_chunk_tokens),
                    page_start=page_start,
                    page_end=page_end,
                    paragraph=len(chunks),
                    extra=extra,
                )

                chunks.append(chunk)

                # Start new chunk with overlap
                # Keep last `overlap` tokens for context
                if self.overlap > 0 and len(current_chunk_tokens) > self.overlap:
                    # Calculate how many sentences to keep for overlap
                    overlap_tokens = current_chunk_tokens[-self.overlap:]
                    overlap_sentences = []
                    token_count = 0

                    # Work backwards through sentences to build overlap
                    for sent in reversed(current_chunk_sentences):
                        sent_tokens = self.tokenize(sent)
                        if token_count + len(sent_tokens) <= self.overlap:
                            overlap_sentences.insert(0, sent)
                            token_count += len(sent_tokens)
                        else:
                            break

                    current_chunk_sentences = overlap_sentences
                    current_chunk_tokens = []
                    for sent in overlap_sentences:
                        current_chunk_tokens.extend(self.tokenize(sent))

                    # Adjust start char to beginning of overlap
                    current_start_char = chunk_end_char - len(" ".join(overlap_sentences))
                else:
                    # No overlap
                    current_chunk_sentences = []
                    current_chunk_tokens = []
                    current_start_char = chunk_end_char

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_tokens.extend(sentence_tokens)
            char_offset += len(sentence) + 1  # +1 for space

        # Create final chunk if there's remaining content
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_end_char = len(text)

            chunk = self._create_chunk(
                text=chunk_text,
                doc_id=doc_id,
                section_path=section_path,
                start_char=current_start_char,
                end_char=chunk_end_char,
                token_count=len(current_chunk_tokens),
                page_start=page_start,
                page_end=page_end,
                paragraph=len(chunks),
                extra=extra,
            )

            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks from {len(sentences)} sentences "
            f"(avg {len(text) / len(chunks):.0f} chars/chunk)"
        )

        return chunks

    def _create_chunk(
        self,
        text: str,
        doc_id: str,
        section_path: str,
        start_char: int,
        end_char: int,
        token_count: int,
        page_start: Optional[int],
        page_end: Optional[int],
        paragraph: int,
        extra: Optional[dict],
    ) -> Chunk:
        """
        Create a Chunk object with stable ID.

        Args:
            text: Chunk text
            doc_id: Document ID
            section_path: Section path
            start_char: Start character offset
            end_char: End character offset
            token_count: Number of tokens
            page_start: Starting page
            page_end: Ending page
            paragraph: Paragraph index
            extra: Extra metadata

        Returns:
            Chunk object
        """
        # Generate stable chunk ID using UUID v5
        chunk_id = chunk_uuid5(doc_id, section_path, start_char)

        # Compute section hash
        sec_hash = section_hash(text, section_path)

        # Create chunk
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            section_path=section_path,
            section_hash=sec_hash,
            page_start=page_start,
            page_end=page_end,
            paragraph=paragraph,
            start_char=start_char,
            end_char=end_char,
            text=text,
            token_count=token_count,
            embedding_hash=None,  # Set later when embedding is generated
            embedding_model=None,  # Set later
            extra=extra or {},
        )

        return chunk


class CharacterChunker:
    """
    Simple character-based chunker (fallback).

    Splits text by character count with overlap.
    Less sophisticated than TokenAwareChunker.
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        overlap: int = 200,
    ):
        """
        Initialize character chunker.

        Args:
            chunk_size: Chunk size in characters
            overlap: Overlap in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        section_path: str = "Document",
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        Chunk document by character count.

        Args:
            text: Document text
            doc_id: Document ID
            section_path: Section path
            page_start: Starting page
            page_end: Ending page
            extra: Extra metadata

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        chunks = []
        start = 0

        while start < len(text):
            # Extract chunk
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # Estimate token count (rough: 1 token ≈ 4 chars)
            token_count = len(chunk_text) // 4

            # Generate chunk ID
            chunk_id = chunk_uuid5(doc_id, section_path, start)
            sec_hash = section_hash(chunk_text, section_path)

            # Create chunk
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                section_path=section_path,
                section_hash=sec_hash,
                page_start=page_start,
                page_end=page_end,
                paragraph=len(chunks),
                start_char=start,
                end_char=end,
                text=chunk_text,
                token_count=token_count,
                embedding_hash=None,
                embedding_model=None,
                extra=extra or {},
            )

            chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - self.overlap

            # Avoid infinite loop
            if start >= end:
                start = end

        logger.info(f"Created {len(chunks)} character-based chunks")

        return chunks


def get_chunker(
    config: Optional[RAGConfig] = None,
) -> TokenAwareChunker:
    """
    Get chunker based on configuration.

    Args:
        config: RAG configuration

    Returns:
        Chunker instance
    """
    config = config or get_config()

    if config.chunking_strategy == "token_aware":
        return TokenAwareChunker(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            config=config,
        )
    elif config.chunking_strategy == "character":
        return CharacterChunker(
            chunk_size=config.chunk_size * 4,  # Rough conversion: tokens -> chars
            overlap=config.chunk_overlap * 4,
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy: {config.chunking_strategy}. "
            f"Supported: token_aware, character"
        )
