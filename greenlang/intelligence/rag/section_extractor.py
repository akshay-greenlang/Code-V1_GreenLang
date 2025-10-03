"""
Extract section hierarchies from climate documents.

CRITICAL: Climate standards have complex hierarchical structures that must be
preserved for proper citation and regulatory compliance.

Examples:
- IPCC AR6: "WG3 > Chapter 6 > Box 6.2 > Figure 6.3a"
- GHG Protocol: "Appendix E > Table E.1"
- ISO 14064: "5.2.3 > Quantification of GHG emissions"

Key features:
- Handle IPCC AR6 complexity (working groups, chapters, boxes, figures)
- Handle GHG Protocol structure (chapters, appendices, tables)
- Handle ISO structure (hierarchical numbering)
- Regex patterns for common standards
- PDF header extraction for section detection
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Pattern
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    logger.warning("pdfplumber not available. Install with: pip install pdfplumber")
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    logger.warning("PyPDF2 not available. Install with: pip install PyPDF2")
    PYPDF2_AVAILABLE = False


@dataclass
class SectionMatch:
    """
    Represents a matched section in a document.
    """
    section_type: str  # 'chapter', 'box', 'figure', 'table', 'appendix', etc.
    section_id: str    # '6', '6.2', 'E', etc.
    section_title: Optional[str] = None
    page_num: Optional[int] = None
    match_text: str = ""


class SectionPathExtractor:
    """
    Extract section hierarchies from climate documents.

    Handles complex structures from IPCC, GHG Protocol, ISO, and other standards.

    Example:
        >>> extractor = SectionPathExtractor()
        >>> # Extract from IPCC AR6 text
        >>> text = "As shown in Chapter 6, Box 6.2, Figure 6.3a..."
        >>> path = extractor.extract_path(text, doc_type='ipcc')
        >>> print(path)  # "Chapter 6 > Box 6.2 > Figure 6.3a"
        >>>
        >>> # Extract from GHG Protocol
        >>> text = "See Appendix E, Table E.1 for emission factors"
        >>> path = extractor.extract_path(text, doc_type='ghg_protocol')
        >>> print(path)  # "Appendix E > Table E.1"
    """

    # Regex patterns for common climate standards
    PATTERNS: Dict[str, Dict[str, Pattern]] = {
        'ipcc': {
            'working_group': re.compile(r'\b(WG|Working Group)\s*([123])\b', re.IGNORECASE),
            'chapter': re.compile(r'\bChapter\s+(\d+)\b', re.IGNORECASE),
            'section': re.compile(r'\bSection\s+([\d\.]+)\b', re.IGNORECASE),
            'box': re.compile(r'\bBox\s+([\d\.]+)\b', re.IGNORECASE),
            'figure': re.compile(r'\bFigure\s+([\d\.]+)([a-z])?\b', re.IGNORECASE),
            'table': re.compile(r'\bTable\s+([\d\.A-Z]+)\b', re.IGNORECASE),
            'annex': re.compile(r'\bAnnex\s+([A-Z]+)\b', re.IGNORECASE),
        },
        'ghg_protocol': {
            'chapter': re.compile(r'\bChapter\s+(\d+)\b', re.IGNORECASE),
            'section': re.compile(r'\bSection\s+([\d\.]+)\b', re.IGNORECASE),
            'appendix': re.compile(r'\bAppendix\s+([A-Z])\b', re.IGNORECASE),
            'table': re.compile(r'\bTable\s+([\d\.A-Z]+)\b', re.IGNORECASE),
            'figure': re.compile(r'\bFigure\s+([\d\.]+)\b', re.IGNORECASE),
            'box': re.compile(r'\bBox\s+([\d\.]+)\b', re.IGNORECASE),
        },
        'iso': {
            'section': re.compile(r'\b(\d+(?:\.\d+)*)\s+([A-Z][^\n]{0,100})\b'),
            'annex': re.compile(r'\bAnnex\s+([A-Z])\b', re.IGNORECASE),
            'table': re.compile(r'\bTable\s+([A-Z]?\.?\d+)\b', re.IGNORECASE),
            'figure': re.compile(r'\bFigure\s+([A-Z]?\.?\d+)\b', re.IGNORECASE),
        },
        'generic': {
            'chapter': re.compile(r'\bChapter\s+(\d+)\b', re.IGNORECASE),
            'section': re.compile(r'\bSection\s+([\d\.]+)\b', re.IGNORECASE),
            'appendix': re.compile(r'\bAppendix\s+([A-Z])\b', re.IGNORECASE),
            'annex': re.compile(r'\bAnnex\s+([A-Z])\b', re.IGNORECASE),
            'table': re.compile(r'\bTable\s+([\d\.A-Z]+)\b', re.IGNORECASE),
            'figure': re.compile(r'\bFigure\s+([\d\.]+)\b', re.IGNORECASE),
        }
    }

    def __init__(self):
        """Initialize section path extractor."""
        pass

    def extract_path(
        self,
        text: str,
        doc_type: str = 'generic',
        max_depth: int = 5
    ) -> str:
        """
        Extract hierarchical section path from text.

        Args:
            text: Input text (can be chunk text or full document)
            doc_type: Document type ('ipcc', 'ghg_protocol', 'iso', 'generic')
            max_depth: Maximum hierarchy depth to extract

        Returns:
            Hierarchical section path (e.g., "Chapter 6 > Box 6.2 > Figure 6.3a")

        Example:
            >>> extractor = SectionPathExtractor()
            >>> text = "As shown in Chapter 6, Section 6.3, Box 6.2..."
            >>> path = extractor.extract_path(text, doc_type='ipcc')
            >>> print(path)  # "Chapter 6 > Section 6.3 > Box 6.2"
        """
        if doc_type not in self.PATTERNS:
            logger.warning(f"Unknown doc_type '{doc_type}', using 'generic'")
            doc_type = 'generic'

        patterns = self.PATTERNS[doc_type]
        matches = []

        # Extract all section references
        for section_type, pattern in patterns.items():
            for match in pattern.finditer(text):
                section_id = match.group(1)
                section_title = None

                # Handle special cases
                if section_type == 'figure' and len(match.groups()) > 1:
                    # Figure with sub-letter (e.g., "Figure 6.3a")
                    sub = match.group(2)
                    if sub:
                        section_id = f"{section_id}{sub}"
                elif section_type == 'section' and doc_type == 'iso' and len(match.groups()) > 1:
                    # ISO section with title (e.g., "5.2.3 Quantification")
                    section_title = match.group(2).strip()

                matches.append(SectionMatch(
                    section_type=section_type,
                    section_id=section_id,
                    section_title=section_title,
                    match_text=match.group(0)
                ))

        if not matches:
            return "Root"

        # Sort matches by position in text
        matches.sort(key=lambda m: text.find(m.match_text))

        # Build hierarchy (limit depth)
        hierarchy = []
        for match in matches[:max_depth]:
            if match.section_title:
                hierarchy.append(f"{match.section_type.capitalize()} {match.section_id} {match.section_title}")
            else:
                hierarchy.append(f"{match.section_type.capitalize()} {match.section_id}")

        return " > ".join(hierarchy) if hierarchy else "Root"

    def extract_from_hierarchy(
        self,
        current_section: str,
        parent_sections: List[str]
    ) -> str:
        """
        Build section path from hierarchy stack.

        Used during PDF parsing when maintaining a section stack.

        Args:
            current_section: Current section (e.g., "6.3.1 Emission Factors")
            parent_sections: Parent sections in order (e.g., ["Chapter 6", "Section 6.3"])

        Returns:
            Full hierarchical path (e.g., "Chapter 6 > Section 6.3 > 6.3.1 Emission Factors")

        Example:
            >>> extractor = SectionPathExtractor()
            >>> path = extractor.extract_from_hierarchy(
            ...     "6.3.1 Emission Factors",
            ...     ["Chapter 6", "Section 6.3"]
            ... )
            >>> print(path)  # "Chapter 6 > Section 6.3 > 6.3.1 Emission Factors"
        """
        hierarchy = parent_sections + [current_section]
        return " > ".join(hierarchy)

    def extract_page_headers(
        self,
        pdf_path: Path,
        header_region: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[int, str]:
        """
        Extract headers from each page for section detection.

        Headers often contain section information (e.g., "Chapter 6: Climate Mitigation").

        Args:
            pdf_path: Path to PDF file
            header_region: Bounding box for header region (x0, y0, x1, y1) in points
                          Default: top 1 inch of page

        Returns:
            Dictionary mapping page number (1-indexed) to header text

        Example:
            >>> extractor = SectionPathExtractor()
            >>> headers = extractor.extract_page_headers(Path("ghg_protocol.pdf"))
            >>> for page, header in headers.items():
            ...     print(f"Page {page}: {header}")
        """
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return {}

        headers = {}

        # Try pdfplumber first (better text extraction)
        if PDFPLUMBER_AVAILABLE:
            headers = self._extract_headers_pdfplumber(pdf_path, header_region)
            if headers:
                return headers

        # Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            headers = self._extract_headers_pypdf2(pdf_path)
            if headers:
                return headers

        logger.warning("No PDF library available for header extraction")
        return {}

    def _extract_headers_pdfplumber(
        self,
        pdf_path: Path,
        header_region: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[int, str]:
        """
        Extract headers using pdfplumber.

        Args:
            pdf_path: Path to PDF
            header_region: Bounding box for header (x0, y0, x1, y1)

        Returns:
            Dictionary mapping page number to header text
        """
        if not PDFPLUMBER_AVAILABLE:
            return {}

        try:
            import pdfplumber

            headers = {}

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Define header region (top 1 inch by default)
                    if header_region is None:
                        bbox = (0, 0, page.width, 72)  # 72 points = 1 inch
                    else:
                        bbox = header_region

                    # Extract text from header region
                    header_crop = page.crop(bbox)
                    header_text = header_crop.extract_text()

                    if header_text:
                        # Clean up header text
                        header_text = header_text.strip()
                        header_text = re.sub(r'\s+', ' ', header_text)
                        headers[page_num] = header_text

            logger.info(f"Extracted headers from {len(headers)} pages using pdfplumber")
            return headers

        except Exception as e:
            logger.error(f"pdfplumber header extraction failed: {e}")
            return {}

    def _extract_headers_pypdf2(
        self,
        pdf_path: Path
    ) -> Dict[int, str]:
        """
        Extract headers using PyPDF2.

        Note: PyPDF2 doesn't support region-based extraction, so this extracts
        first few lines from each page.

        Args:
            pdf_path: Path to PDF

        Returns:
            Dictionary mapping page number to header text
        """
        if not PYPDF2_AVAILABLE:
            return {}

        try:
            import PyPDF2

            headers = {}

            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)

                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()

                    # Take first 2 lines as header
                    lines = text.split('\n')[:2]
                    header_text = ' '.join(lines).strip()

                    if header_text:
                        headers[page_num + 1] = header_text

            logger.info(f"Extracted headers from {len(headers)} pages using PyPDF2")
            return headers

        except Exception as e:
            logger.error(f"PyPDF2 header extraction failed: {e}")
            return {}

    def detect_section_type(self, text: str) -> str:
        """
        Detect document type from text content.

        Looks for characteristic patterns to identify IPCC, GHG Protocol, ISO, etc.

        Args:
            text: Document text sample (first few pages recommended)

        Returns:
            Document type ('ipcc', 'ghg_protocol', 'iso', 'generic')

        Example:
            >>> extractor = SectionPathExtractor()
            >>> text = "IPCC AR6 WG3 Chapter 6..."
            >>> doc_type = extractor.detect_section_type(text)
            >>> print(doc_type)  # "ipcc"
        """
        text_lower = text.lower()

        # Check for IPCC
        if any(keyword in text_lower for keyword in ['ipcc', 'intergovernmental panel', 'working group']):
            if re.search(r'\bwg\s*[123]\b', text_lower) or 'ar6' in text_lower or 'ar5' in text_lower:
                return 'ipcc'

        # Check for GHG Protocol
        if any(keyword in text_lower for keyword in ['ghg protocol', 'greenhouse gas protocol', 'wri/wbcsd']):
            return 'ghg_protocol'

        # Check for ISO
        if re.search(r'\biso\s+\d{4,5}', text_lower):
            return 'iso'

        # Default to generic
        return 'generic'

    def normalize_section_path(self, section_path: str) -> str:
        """
        Normalize section path for consistent formatting.

        - Remove extra whitespace
        - Standardize separators
        - Title case section types

        Args:
            section_path: Raw section path

        Returns:
            Normalized section path

        Example:
            >>> extractor = SectionPathExtractor()
            >>> path = "chapter 6  >  box 6.2 >figure 6.3a"
            >>> normalized = extractor.normalize_section_path(path)
            >>> print(normalized)  # "Chapter 6 > Box 6.2 > Figure 6.3a"
        """
        # Split by separator
        parts = section_path.split('>')

        # Normalize each part
        normalized_parts = []
        for part in parts:
            # Remove extra whitespace
            part = ' '.join(part.split())

            # Title case section types
            part = self._title_case_section_types(part)

            normalized_parts.append(part)

        return ' > '.join(normalized_parts)

    def _title_case_section_types(self, text: str) -> str:
        """
        Title case known section types.

        Args:
            text: Section text

        Returns:
            Text with section types title-cased
        """
        section_types = [
            'chapter', 'section', 'appendix', 'annex', 'box', 'figure', 'table',
            'working group', 'wg'
        ]

        for section_type in section_types:
            # Replace with title case (case-insensitive)
            text = re.sub(
                rf'\b{section_type}\b',
                section_type.title(),
                text,
                flags=re.IGNORECASE
            )

        return text

    def get_section_anchor(self, section_path: str) -> str:
        """
        Generate URL anchor from section path.

        Converts section path to URL-safe anchor for citations.

        Args:
            section_path: Section path (e.g., "Chapter 6 > Box 6.2")

        Returns:
            URL anchor (e.g., "chapter_6_box_6_2")

        Example:
            >>> extractor = SectionPathExtractor()
            >>> anchor = extractor.get_section_anchor("Chapter 6 > Box 6.2 > Figure 6.3a")
            >>> print(anchor)  # "chapter_6_box_6_2_figure_6_3a"
        """
        # Remove '>' separators
        anchor = section_path.replace('>', ' ')

        # Lowercase
        anchor = anchor.lower()

        # Replace spaces and special chars with underscore
        anchor = re.sub(r'[^\w]+', '_', anchor)

        # Remove leading/trailing underscores
        anchor = anchor.strip('_')

        return anchor
