# -*- coding: utf-8 -*-
"""
Tests for IntakeAgent
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from greenlang.agents.templates import IntakeAgent, DataFormat


class TestIntakeAgent:
    """Test IntakeAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization."""
        agent = IntakeAgent()
        assert agent is not None
        assert agent.schema == {}

    @pytest.mark.asyncio
    async def test_ingest_csv_from_dataframe(self):
        """Test ingesting CSV data from DataFrame."""
        agent = IntakeAgent()

        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "value": [100, 200]
        })

        result = await agent.ingest(data=df, format=DataFormat.CSV)

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_ingest_with_validation(self):
        """Test ingestion with validation."""
        schema = {
            "required": ["name", "value"],
            "types": {
                "name": "string",
                "value": "number"
            }
        }
        agent = IntakeAgent(schema=schema)

        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "value": [100, 200]
        })

        result = await agent.ingest(data=df, format=DataFormat.CSV, validate=True)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_ingest_missing_required_columns(self):
        """Test ingestion with missing required columns."""
        schema = {
            "required": ["name", "value", "missing_col"],
        }
        agent = IntakeAgent(schema=schema)

        df = pd.DataFrame({
            "name": ["Alice"],
            "value": [100]
        })

        result = await agent.ingest(data=df, format=DataFormat.CSV, validate=True)

        # Should have validation issues
        assert len(result.validation_issues) > 0

    @pytest.mark.asyncio
    async def test_ingest_csv_file(self):
        """Test ingesting from CSV file."""
        agent = IntakeAgent()

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("name,value\n")
            f.write("Alice,100\n")
            f.write("Bob,200\n")
            temp_path = f.name

        try:
            result = await agent.ingest(file_path=temp_path, format=DataFormat.CSV)

            assert result.success is True
            assert result.rows_read == 2

        finally:
            Path(temp_path).unlink()

    def test_get_stats(self):
        """Test getting agent statistics."""
        agent = IntakeAgent()
        stats = agent.get_stats()

        assert "total_intakes" in stats
        assert stats["total_intakes"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
