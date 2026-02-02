# -*- coding: utf-8 -*-
"""PDF Exporter - GL-VCCI Scope 3 Platform v1.0.0"""
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFExporter:
    """Exports reports to PDF format using HTML templates."""

    def export(self, content: Dict[str, Any], html_content: str, output_path: str) -> str:
        """Export to PDF."""
        logger.info(f"Exporting to PDF: {output_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            # Try using weasyprint if available
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(output_path)
            logger.info(f"PDF generated using WeasyPrint: {output_path}")
        except ImportError:
            # Fallback: save as HTML
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.warning(f"WeasyPrint not available, saved as HTML: {html_path}")
            return html_path

        return output_path

__all__ = ["PDFExporter"]
