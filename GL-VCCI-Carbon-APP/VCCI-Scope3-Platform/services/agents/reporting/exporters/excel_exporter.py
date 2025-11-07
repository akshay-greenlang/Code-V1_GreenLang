"""Excel Exporter - GL-VCCI Scope 3 Platform v1.0.0"""
import logging
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class ExcelExporter:
    """Exports reports to Excel format."""

    def export(self, content: Dict[str, Any], tables: Dict[str, pd.DataFrame],
               output_path: str) -> str:
        """Export to Excel workbook."""
        logger.info(f"Exporting to Excel: {output_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                "Metric": ["Total Emissions", "Scope 1", "Scope 2", "Scope 3"],
                "Value (tCO2e)": [
                    content.get("total_emissions", 0),
                    content.get("scope1", 0),
                    content.get("scope2", 0),
                    content.get("scope3", 0),
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

            # Additional sheets from tables
            for sheet_name, df in tables.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        logger.info(f"Excel export complete: {output_path}")
        return output_path

__all__ = ["ExcelExporter"]
