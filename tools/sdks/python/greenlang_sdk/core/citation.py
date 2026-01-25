"""
Citation tracking for regulatory compliance.

Ensures all emissions factors and regulatory frameworks are properly cited.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class CitationRecord(BaseModel):
    """Single citation for an emission factor or regulatory source."""

    source: str = Field(..., description="Source identifier (e.g., 'DEFRA 2024', 'GHG Protocol')")
    url: Optional[HttpUrl] = Field(None, description="URL to source document")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: Optional[str] = Field(None, description="Source version (e.g., 'AR6', '2024')")
    page: Optional[str] = Field(None, description="Page number or section")
    data_hash: Optional[str] = Field(None, description="SHA-256 hash of cited data")

    class Config:
        json_schema_extra = {
            "example": {
                "source": "DEFRA 2024 Emission Factors",
                "url": "https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
                "version": "2024",
                "page": "Table 1.2",
                "data_hash": "a1b2c3d4e5f6..."
            }
        }


class CitationTracker:
    """Tracks all citations during agent execution."""

    def __init__(self):
        self.citations: List[CitationRecord] = []

    def add_citation(
        self,
        source: str,
        url: Optional[str] = None,
        version: Optional[str] = None,
        page: Optional[str] = None,
        data_hash: Optional[str] = None
    ) -> CitationRecord:
        """Add a citation record."""
        citation = CitationRecord(
            source=source,
            url=url,
            version=version,
            page=page,
            data_hash=data_hash
        )
        self.citations.append(citation)
        return citation

    def get_all_citations(self) -> List[CitationRecord]:
        """Get all citations."""
        return self.citations.copy()

    def get_unique_sources(self) -> List[str]:
        """Get unique list of cited sources."""
        return list(set(c.source for c in self.citations))

    def citation_coverage(self) -> float:
        """Calculate citation coverage (% of operations with citations)."""
        # This would be calculated based on tool calls with citations
        # For now, return count
        return len(self.citations)
