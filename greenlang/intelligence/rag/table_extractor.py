"""
Extract emission factor tables from climate PDFs.

CRITICAL: Tables in climate/GHG standards contain critical emission factors,
conversion coefficients, and compliance thresholds. This module extracts tables
with full structure preservation for regulatory compliance.

Key features:
- Extract tables from PDFs using Camelot or Tabula
- Preserve row/column structure
- Extract units and footnotes
- Store as structured JSON in chunk metadata
- Create embeddings for table search

Dependencies:
    pip install camelot-py[cv] tabula-py

Note:
    Camelot requires ghostscript: https://www.ghostscript.com/download/gsdnld.html
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

logger = logging.getLogger(__name__)

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    logger.warning("Camelot not available. Install with: pip install camelot-py[cv]")
    CAMELOT_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    logger.warning("Tabula not available. Install with: pip install tabula-py")
    TABULA_AVAILABLE = False


class ClimateTableExtractor:
    """
    Extract emission factor tables from climate PDFs.

    Uses Camelot for lattice/stream table detection and Tabula as fallback.

    Example:
        >>> extractor = ClimateTableExtractor()
        >>> # Extract all tables from GHG Protocol Chapter 7
        >>> tables = extractor.extract_tables(
        ...     Path("ghg_protocol.pdf"),
        ...     page_num=45
        ... )
        >>> for table in tables:
        ...     print(f"Table: {table['caption']}")
        ...     print(f"Rows: {len(table['rows'])}, Columns: {len(table['headers'])}")
        ...     print(f"Units: {table['units']}")
    """

    def __init__(
        self,
        prefer_camelot: bool = True,
        flavor: str = 'lattice'
    ):
        """
        Initialize table extractor.

        Args:
            prefer_camelot: Prefer Camelot over Tabula (better quality)
            flavor: Camelot flavor ('lattice' or 'stream')
                   - lattice: For tables with borders
                   - stream: For tables without borders
        """
        self.prefer_camelot = prefer_camelot and CAMELOT_AVAILABLE
        self.flavor = flavor

        if not CAMELOT_AVAILABLE and not TABULA_AVAILABLE:
            logger.error(
                "Neither Camelot nor Tabula available. "
                "Install with: pip install camelot-py[cv] tabula-py"
            )

    def extract_tables(
        self,
        pdf_path: Path,
        page_num: Optional[int] = None,
        pages: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract tables from PDF.

        Args:
            pdf_path: Path to PDF file
            page_num: Single page number (1-indexed), or None for all pages
            pages: Page range string (e.g., '1-3,5,7-9') for Camelot

        Returns:
            List of table dictionaries with structure:
            {
                'caption': str,
                'headers': List[str],
                'rows': List[List[str]],
                'units': Dict[str, str],  # column → unit
                'notes': List[str],       # footnotes
                'page': int,              # page number
                'bbox': Tuple[float, float, float, float]  # bounding box
            }

        Example:
            >>> extractor = ClimateTableExtractor()
            >>> tables = extractor.extract_tables(Path("ghg_protocol.pdf"), page_num=45)
            >>> table = tables[0]
            >>> print(f"Table: {table['caption']}")
            >>> print(f"Headers: {table['headers']}")
            >>> for row in table['rows']:
            ...     print(row)
        """
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return []

        # Determine page specification
        page_spec = pages
        if page_spec is None and page_num is not None:
            page_spec = str(page_num)
        elif page_spec is None:
            page_spec = 'all'

        # Try Camelot first (better quality)
        if self.prefer_camelot:
            tables = self._extract_with_camelot(pdf_path, page_spec)
            if tables:
                logger.info(f"Extracted {len(tables)} tables with Camelot from {pdf_path.name}")
                return tables
            else:
                logger.warning("Camelot extraction failed, trying Tabula")

        # Fallback to Tabula
        if TABULA_AVAILABLE:
            tables = self._extract_with_tabula(pdf_path, page_spec)
            logger.info(f"Extracted {len(tables)} tables with Tabula from {pdf_path.name}")
            return tables
        else:
            logger.error("No table extraction library available")
            return []

    def _extract_with_camelot(
        self,
        pdf_path: Path,
        pages: str
    ) -> List[Dict]:
        """
        Extract tables using Camelot.

        Args:
            pdf_path: Path to PDF
            pages: Page specification ('1', '1-3', 'all')

        Returns:
            List of table dictionaries
        """
        if not CAMELOT_AVAILABLE:
            return []

        try:
            # Extract tables with Camelot
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=pages,
                flavor=self.flavor,
                strip_text='\n'
            )

            result = []
            for i, table in enumerate(tables):
                # Get table data
                df = table.df
                page_num = table.page

                # Extract headers (first row)
                headers = df.iloc[0].tolist() if len(df) > 0 else []

                # Extract rows (remaining rows)
                rows = df.iloc[1:].values.tolist() if len(df) > 1 else []

                # Extract caption (try to find from preceding text)
                caption = f"Table {i+1} on page {page_num}"

                # Extract units from headers (e.g., "CO2 (kg/unit)")
                units = self._extract_units_from_headers(headers)

                # Extract footnotes (look for rows with '*', '†', etc.)
                notes = self._extract_footnotes(rows)

                # Get bounding box
                bbox = (
                    float(table._bbox[0]),
                    float(table._bbox[1]),
                    float(table._bbox[2]),
                    float(table._bbox[3])
                ) if hasattr(table, '_bbox') else (0, 0, 0, 0)

                table_dict = {
                    'caption': caption,
                    'headers': headers,
                    'rows': rows,
                    'units': units,
                    'notes': notes,
                    'page': page_num,
                    'bbox': bbox,
                    'accuracy': float(table.accuracy) if hasattr(table, 'accuracy') else 0.0
                }

                result.append(table_dict)

            return result

        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}")
            return []

    def _extract_with_tabula(
        self,
        pdf_path: Path,
        pages: str
    ) -> List[Dict]:
        """
        Extract tables using Tabula.

        Args:
            pdf_path: Path to PDF
            pages: Page specification ('1', '1-3', 'all')

        Returns:
            List of table dictionaries
        """
        if not TABULA_AVAILABLE:
            return []

        try:
            # Extract tables with Tabula
            if pages == 'all':
                page_list = 'all'
            else:
                # Convert page spec to list
                page_list = pages

            dfs = tabula.read_pdf(
                str(pdf_path),
                pages=page_list,
                multiple_tables=True,
                pandas_options={'header': None}
            )

            result = []
            for i, df in enumerate(dfs):
                if df.empty:
                    continue

                # Extract headers (first row)
                headers = df.iloc[0].tolist() if len(df) > 0 else []

                # Extract rows (remaining rows)
                rows = df.iloc[1:].values.tolist() if len(df) > 1 else []

                # Extract caption
                caption = f"Table {i+1}"

                # Extract units
                units = self._extract_units_from_headers(headers)

                # Extract footnotes
                notes = self._extract_footnotes(rows)

                table_dict = {
                    'caption': caption,
                    'headers': headers,
                    'rows': rows,
                    'units': units,
                    'notes': notes,
                    'page': 0,  # Tabula doesn't provide page info easily
                    'bbox': (0, 0, 0, 0)
                }

                result.append(table_dict)

            return result

        except Exception as e:
            logger.error(f"Tabula extraction failed: {e}")
            return []

    def _extract_units_from_headers(self, headers: List[str]) -> Dict[str, str]:
        """
        Extract units from table headers.

        Example:
            "CO2 (kg/unit)" → {'CO2': 'kg/unit'}
            "Energy (MJ)" → {'Energy': 'MJ'}

        Args:
            headers: List of header strings

        Returns:
            Dictionary mapping column name to unit
        """
        import re
        units = {}

        for header in headers:
            if not isinstance(header, str):
                continue

            # Match patterns like "Name (unit)" or "Name [unit]"
            match = re.search(r'(.+?)\s*[\(\[](.+?)[\)\]]', header)
            if match:
                col_name = match.group(1).strip()
                unit = match.group(2).strip()
                units[col_name] = unit

        return units

    def _extract_footnotes(self, rows: List[List]) -> List[str]:
        """
        Extract footnotes from table rows.

        Looks for rows containing footnote markers: *, †, ‡, §, etc.

        Args:
            rows: List of table rows

        Returns:
            List of footnote strings
        """
        footnote_markers = ['*', '†', '‡', '§', '¶', '**', 'a)', 'b)', 'c)']
        notes = []

        for row in rows:
            # Convert row to string
            row_str = ' '.join(str(cell) for cell in row if cell)

            # Check if row contains footnote marker
            for marker in footnote_markers:
                if marker in row_str:
                    notes.append(row_str.strip())
                    break

        return notes

    def embed_table(
        self,
        table_struct: Dict,
        embedding_model: Optional[any] = None
    ) -> Dict:
        """
        Create embeddings for table search.

        Generates embeddings for:
        1. Table caption
        2. Each row (concatenated)
        3. Column headers

        Args:
            table_struct: Table dictionary from extract_tables()
            embedding_model: Optional embedding model (SentenceTransformer, etc.)

        Returns:
            Dictionary with embeddings added:
            {
                'caption_embedding': List[float],
                'row_embeddings': List[List[float]],
                'header_embedding': List[float]
            }

        Example:
            >>> from sentence_transformers import SentenceTransformer
            >>> model = SentenceTransformer('all-MiniLM-L6-v2')
            >>> extractor = ClimateTableExtractor()
            >>> tables = extractor.extract_tables(Path("ghg_protocol.pdf"))
            >>> table_with_embeddings = extractor.embed_table(tables[0], model)
        """
        if embedding_model is None:
            logger.warning("No embedding model provided, skipping embedding")
            return table_struct

        try:
            # 1. Embed caption
            caption_text = table_struct['caption']
            caption_embedding = embedding_model.encode(caption_text).tolist()

            # 2. Embed headers
            headers_text = " | ".join(table_struct['headers'])
            header_embedding = embedding_model.encode(headers_text).tolist()

            # 3. Embed each row
            row_embeddings = []
            for row in table_struct['rows']:
                row_text = " | ".join(str(cell) for cell in row if cell)
                row_embedding = embedding_model.encode(row_text).tolist()
                row_embeddings.append(row_embedding)

            # Add embeddings to structure
            table_struct['caption_embedding'] = caption_embedding
            table_struct['header_embedding'] = header_embedding
            table_struct['row_embeddings'] = row_embeddings

            logger.info(
                f"Generated embeddings for table '{table_struct['caption']}' "
                f"({len(row_embeddings)} rows)"
            )

            return table_struct

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return table_struct

    def table_to_markdown(self, table_struct: Dict) -> str:
        """
        Convert table structure to markdown format.

        Useful for storing tables in chunk text for LLM context.

        Args:
            table_struct: Table dictionary from extract_tables()

        Returns:
            Markdown table string

        Example:
            >>> extractor = ClimateTableExtractor()
            >>> tables = extractor.extract_tables(Path("ghg_protocol.pdf"))
            >>> md = extractor.table_to_markdown(tables[0])
            >>> print(md)
            # Table: Emission Factors
            | Fuel Type | CO2 (kg/unit) | CH4 (kg/unit) |
            |-----------|---------------|---------------|
            | Natural Gas | 2.0 | 0.001 |
            | Coal | 3.5 | 0.002 |
        """
        lines = []

        # Add caption
        lines.append(f"# Table: {table_struct['caption']}")
        lines.append("")

        # Add headers
        headers = table_struct['headers']
        if headers:
            lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            lines.append("|" + "|".join("---" for _ in headers) + "|")

        # Add rows
        for row in table_struct['rows']:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        # Add units (if present)
        if table_struct['units']:
            lines.append("")
            lines.append("**Units:**")
            for col, unit in table_struct['units'].items():
                lines.append(f"- {col}: {unit}")

        # Add notes (if present)
        if table_struct['notes']:
            lines.append("")
            lines.append("**Notes:**")
            for note in table_struct['notes']:
                lines.append(f"- {note}")

        return "\n".join(lines)

    def table_to_json(self, table_struct: Dict) -> str:
        """
        Convert table structure to JSON string.

        Args:
            table_struct: Table dictionary from extract_tables()

        Returns:
            JSON string

        Example:
            >>> extractor = ClimateTableExtractor()
            >>> tables = extractor.extract_tables(Path("ghg_protocol.pdf"))
            >>> json_str = extractor.table_to_json(tables[0])
            >>> print(json_str)
        """
        # Remove embeddings for cleaner JSON (they're too large)
        table_copy = table_struct.copy()
        table_copy.pop('caption_embedding', None)
        table_copy.pop('header_embedding', None)
        table_copy.pop('row_embeddings', None)

        return json.dumps(table_copy, indent=2)
