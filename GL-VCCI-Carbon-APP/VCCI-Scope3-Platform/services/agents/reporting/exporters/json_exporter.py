"""JSON Exporter - GL-VCCI Scope 3 Platform v1.0.0"""
import json
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class JSONExporter:
    """Exports reports to JSON format."""

    def export(self, content: Dict[str, Any], output_path: str) -> str:
        """Export to JSON."""
        logger.info(f"Exporting to JSON: {output_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"JSON export complete: {output_path}")
        return output_path

__all__ = ["JSONExporter"]
